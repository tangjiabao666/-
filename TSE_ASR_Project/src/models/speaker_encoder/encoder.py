#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSE-ASR 项目 —— 轻量级声纹提取网络 (LightSpeakerEncoder)。

本模块实现了一个基于简化 ECAPA-TDNN 思想的轻量级 Speaker Encoder，
用于从唤醒音频 (enroll_wavs) 中提取固定维度的目标说话人嵌入向量。

网络结构概览
------------
::

    Raw Waveform [B, 1, T]
           │
           ▼
    ┌─────────────────────┐
    │  MelSpectrogram     │  → Fbank 特征 [B, n_mels, T_feat]
    │  + Log Compression  │
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  Conv1D Frontend    │  → [B, C, T_feat]    (1×1 conv 升维)
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  SE-Res2Block ×3    │  → 多层残差卷积 + SE 通道注意力
    │  (多尺度特征融合)    │
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  Multi-layer Cat    │  → 拼接 3 层输出 [B, 3*C, T]
    │  + 1×1 Fusion       │  → 降维融合 [B, C, T]
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  Attentive Stats    │  → 加权均值 + 加权标准差
    │  Pooling (ASP)      │     [B, 2*C]
    │  + Mask (lengths)   │     padding 帧不参与统计
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  Linear + BN        │  → Speaker Embedding [B, embedding_dim]
    │  + L2 Normalize     │
    └─────────────────────┘

参考论文
--------
- ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN
  Based Speaker Verification (Desplanques et al., Interspeech 2020)
- Res2Net: A New Multi-scale Backbone Architecture (Gao et al., TPAMI 2021)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ====================================================================
# 基础组件 1：Same Padding 卷积
# ====================================================================

class Conv1dSame(nn.Module):
    """带有 'same' 填充的 Conv1d，保证输入输出时间维度一致。

    在 TDNN 架构中，各层需要保持时间分辨率不变，以便残差连接相加。
    本模块通过计算 kernel_size 和 dilation 自动推导出 'same' padding，
    确保卷积操作不会改变时间维度长度。

    Parameters
    ----------
    in_channels : int
        输入通道数。
    out_channels : int
        输出通道数。
    kernel_size : int
        卷积核大小。
    dilation : int, optional
        膨胀率，默认 1。膨胀率 >1 可增大感受野而不增加参数量。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        # 计算单侧填充量，实现 'same' padding
        # 公式: padding = (kernel_size - 1) * dilation / 2
        self.padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入张量，shape [B, C, T]。

        Returns
        -------
        torch.Tensor
            输出张量，shape [B, out_C, T]（时间维度不变）。
        """
        return self.conv(x)


# ====================================================================
# 基础组件 2：Squeeze-and-Excitation 通道注意力
# ====================================================================

class SELayer(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    对通道维度做：
        全局平均池化 (Squeeze) → FC → ReLU → FC → Sigmoid (Excitation)
    得到每个通道的重要性权重，再乘回原始特征实现通道注意力加权。

    这种机制让网络自动学习"哪些通道对说话人区分更有用"，
    是 ECAPA-TDNN 相比传统 x-vector 的关键提升之一。

    Parameters
    ----------
    channels : int
        输入/输出通道数。
    reduction : int, optional
        瓶颈压缩比例，默认 8。值越小 SE 模块参数量越少。
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        bottleneck = max(channels // reduction, 1)
        # 使用 Conv1d(kernel_size=1) 等价于全连接层，但可直接作用在 [B, C, T] 上
        self.fc1 = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.fc2 = nn.Conv1d(bottleneck, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入张量，shape [B, C, T]。

        Returns
        -------
        torch.Tensor
            通道注意力加权后的输出，shape [B, C, T]。
        """
        # Squeeze：对时间维度做全局平均池化，得到通道级统计量
        se = x.mean(dim=2, keepdim=True)  # [B, C, 1]
        # Excitation：两个 1×1 卷积 + 非线性
        se = F.relu(self.fc1(se))         # [B, bottleneck, 1]
        se = torch.sigmoid(self.fc2(se))  # [B, C, 1]
        # 通道加权
        return x * se


# ====================================================================
# 基础组件 3：带 SE 注意力的残差卷积块 (SE-ResBlock)
# ====================================================================

class SEBasicBlock(nn.Module):
    """带有 SE 注意力的残差卷积块。

    结构: Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm → SE → Residual Add → ReLU

    这是构建声纹网络的"原子模块"。残差连接确保梯度能跨越多个
    卷积层回传，SE 注意力则动态调节各通道的重要性。

    Parameters
    ----------
    channels : int
        输入/输出通道数（保持不变）。
    kernel_size : int
        卷积核大小。kernel_size=3 常用于局部特征提取。
    dilation : int, optional
        膨胀率，默认 1。
    reduction : int, optional
        SE 瓶颈压缩比例，默认 8。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
        reduction: int = 8,
    ) -> None:
        super().__init__()

        self.conv1 = Conv1dSame(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = Conv1dSame(channels, channels, kernel_size, dilation=dilation)
        self.se = SELayer(channels, reduction=reduction)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入张量，shape [B, C, T]。

        Returns
        -------
        torch.Tensor
            输出张量，shape [B, C, T]。
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)            # 通道注意力
        out = out + residual          # 残差连接
        return F.relu(out)


# ====================================================================
# 基础组件 4：多尺度 SE-Res2Block
# ====================================================================

class Res2Block(nn.Module):
    """Res2Net 风格的多尺度残差块（简化版）。

    将通道划分为多个子组 (scale 组)，每组依次经过卷积并逐步融合，
    形成层级化的多尺度感受野。配合 SE 注意力，构成本网络的核心
    SE-Res2Block 模块。

    示例 (scale=4, channels=512 → sub_channels=128):
        输入 [B, 512, T]
          ├── splits[0]  [B,128,T] ──→ conv_first ──→ out[0]
          ├── splits[1]  [B,128,T] + out[0] ──→ conv ──→ out[1]
          ├── splits[2]  [B,128,T] + out[1] ──→ conv ──→ out[2]
          └── splits[3]  [B,128,T] + out[2] ──→ conv ──→ out[3]
              → concat → BN → SE → + residual → ReLU

    当通道数不足以分组时（如 channels < scale），自动退化为普通的
    SEBasicBlock，保证模块始终可用。

    Parameters
    ----------
    channels : int
        输入/输出通道数。
    kernel_size : int
        卷积核大小。
    scale : int, optional
        分组数（多尺度分支数），默认 4。
    reduction : int, optional
        SE 瓶颈压缩比例，默认 8。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        scale: int = 4,
        reduction: int = 8,
    ) -> None:
        super().__init__()

        self.scale = scale
        # 每个子分支的通道数
        self.sub_channels = max(channels // scale, 1)
        # 实际分组数可能会因为整除问题而调整
        actual_scale = channels // self.sub_channels
        if actual_scale <= 1:
            # 通道数太少（如 channels=64, scale=4 → sub_channels=16, actual_scale=4 可整除）
            # 但当 channels < scale 时，退化为普通 SEBasicBlock
            self.block = SEBasicBlock(channels, kernel_size, reduction=reduction)
            self.use_res2 = False
        else:
            self.use_res2 = True
            self.scale = actual_scale
            self.sub_channels = channels // actual_scale
            # 第一个分支的卷积（处理初始子通道）
            self.conv_first = Conv1dSame(
                self.sub_channels, self.sub_channels, kernel_size, dilation=1
            )
            # 其余各分支的卷积（每个分支融合前一个分支的输出）
            self.convs = nn.ModuleList([
                Conv1dSame(self.sub_channels, self.sub_channels, kernel_size, dilation=1)
                for _ in range(actual_scale - 1)
            ])
            self.bn = nn.BatchNorm1d(channels)
            self.se = SELayer(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入张量，shape [B, C, T]。

        Returns
        -------
        torch.Tensor
            输出张量，shape [B, C, T]。
        """
        if not self.use_res2:
            return self.block(x)

        residual = x

        # 将通道均匀分为 scale 组
        splits = torch.split(x, self.sub_channels, dim=1)  # scale 个 [B, sub_C, T]

        # 第一个分支直接卷积
        out_splits = [self.conv_first(splits[0])]

        # 后续分支：输入 = 前一分支输出 + 当前分支原始输入 → 卷积
        for i, conv in enumerate(self.convs):
            sp_input = splits[i + 1] + out_splits[i]
            out_splits.append(conv(sp_input))

        # 拼接所有分支输出
        out = torch.cat(out_splits, dim=1)  # [B, C, T]
        out = self.bn(out)
        out = self.se(out)
        out = out + residual
        return F.relu(out)


# ====================================================================
# 基础组件 5：注意力统计池化 (Attentive Statistics Pooling, ASP)
# ====================================================================

class AttentiveStatisticsPooling(nn.Module):
    """注意力统计池化 (Attentive Statistics Pooling, ASP)。

    这是将变长的帧级特征压缩为固定维度句子级表征的关键模块。

    工作原理:
        1. 对每一帧计算一个标量注意力分数 (通过 1×1 卷积)
        2. 对注意力分数做 softmax 得到权重分布
        3. 用权重对帧级特征做加权求和，得到加权均值 (μ)
        4. 用权重对帧级特征平方做加权求和，计算加权标准差 (σ)
        5. 拼接 [μ, σ] 作为句子级表征

    **关键设计**: 结合 ``lengths`` 参数，将 padding 帧的注意力分数置为 -inf，
    使其在 softmax 后权重趋近于 0，从而这些静音帧不参与声纹统计。
    这是确保变长 audio 被正确处理的核心机制。

    Parameters
    ----------
    channels : int
        输入特征通道数。
    attention_channels : int, optional
        注意力瓶颈层通道数，默认 128。瓶颈层减少参数量并防止过拟合。
    """

    def __init__(self, channels: int, attention_channels: int = 128) -> None:
        super().__init__()

        # 注意力计算: channels → bottleneck → 1 (标量注意力分数)
        self.attention = nn.Sequential(
            Conv1dSame(channels, attention_channels, kernel_size=1),  # 1×1 conv 降维
            nn.ReLU(),
            nn.Conv1d(attention_channels, 1, kernel_size=1),           # 输出 1 维分数
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入特征，shape [B, C, T]。其中部分帧可能是 padding 产生的静音帧。
        lengths : torch.Tensor
            各样本的真实帧数（不含 padding），shape [B]。
            用于生成 mask，将 padding 区域排除在统计池化之外。

        Returns
        -------
        torch.Tensor
            池化后的句子级表征，shape [B, 2*C]（均值拼接标准差）。
        """
        # Step 1: 计算每帧的注意力分数
        attn_scores = self.attention(x)  # [B, 1, T]

        # Step 2: 生成 padding mask，将无效帧的注意力分数置为 -inf
        max_len = x.shape[2]
        # mask[i, t] = True 表示样本 i 的第 t 帧是有效的
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).float()  # [B, 1, T], 1=有效, 0=padding
        # 用 -1e4 填充（float16 最大 ~65504，-1e9 会 overflow）
        # 这个值在 softmax 后权重同样趋近于 0
        attn_scores = attn_scores.masked_fill(mask == 0, -1e4)

        # Step 3: Softmax 得到每帧的归一化注意力权重
        attn_weights = F.softmax(attn_scores, dim=2)  # [B, 1, T]

        # Step 4: 加权均值 μ = Σ(w_t · x_t)
        mean = torch.sum(x * attn_weights, dim=2)  # [B, C]

        # Step 5: 加权标准差 σ = sqrt( Σ(w_t · x_t²) - μ² )
        mean_sq = torch.sum((x ** 2) * attn_weights, dim=2)  # E[x²]
        std = torch.sqrt(torch.clamp(mean_sq - mean ** 2, min=1e-9))  # [B, C]

        # Step 6: 拼接均值与标准差
        pooled = torch.cat([mean, std], dim=1)  # [B, 2*C]

        return pooled


# ====================================================================
# 主网络：LightSpeakerEncoder
# ====================================================================

class LightSpeakerEncoder(nn.Module):
    """轻量级声纹提取网络。

    基于简化 ECAPA-TDNN 架构，从原始唤醒音频波形中提取固定维度的
    说话人嵌入向量 (Speaker Embedding)。

    设计思路:
    - MelSpectrogram 内置提取 Fbank 声学特征，避免外部特征提取依赖
    - 3 层 SE-Res2Block 提供多尺度感受野 + 通道注意力
    - 多层特征拼接 (Multi-layer Feature Aggregation) 融合浅层和深层信息
    - ASP 池化将变长序列压缩为固定维度向量，padding mask 确保静音不干扰
    - 输出经 L2 归一化，方便后续与 mixed 音频的嵌入做余弦相似度计算

    Parameters
    ----------
    embedding_dim : int, optional
        输出的说话人嵌入向量维度，默认 256。
    n_mels : int, optional
        Mel 滤波器组数量（Fbank 特征维度），默认 80。
    channels : int, optional
        网络中间层通道数，默认 256（轻量化设计，可根据需要调大）。
    sample_rate : int, optional
        音频采样率，默认 16000。
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        n_mels: int = 80,
        channels: int = 256,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.channels = channels

        # ---------- 音频特征提取层 ----------
        # 25ms 窗口，10ms 帧移 → 符合语音识别与声纹验证标准
        # 轻量化设计：n_fft=512, win_length=400, hop_length=160
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,                    # 32ms @ 16kHz → 频率分辨率 ~31Hz
            win_length=400,               # 25ms 窗口长度
            hop_length=160,               # 10ms 帧移
            f_min=20,                     # 低频截止 (去除直流分量)
            f_max=7600,                   # 8kHz 以内的信息对声纹足够
            n_mels=n_mels,
            power=2.0,                    # 功率谱
        )

        # ---------- Conv1D 前端 ----------
        # 将 n_mels 维 Mel 特征通过 1×1 卷积映射到目标通道数
        self.frontend = nn.Sequential(
            Conv1dSame(n_mels, channels, kernel_size=1),  # 1×1 conv 升维
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        # ---------- SE-Res2Block 层级堆叠 ----------
        # 3 层 SE-Res2Block，每层的 scale=4 提供多尺度感受野
        self.block1 = Res2Block(channels, kernel_size=3, scale=4)
        self.block2 = Res2Block(channels, kernel_size=3, scale=4)
        self.block3 = Res2Block(channels, kernel_size=3, scale=4)

        # ---------- 多层特征拼接与融合 ----------
        # ECAPA-TDNN 核心创新：将不同层的输出拼接在一起
        # 浅层捕捉局部纹理，深层捕捉全局语义，融合后信息更丰富
        self.cat_channels = channels * 3  # 拼接 block1 + block2 + block3

        # 1×1 卷积降维融合，将 3*channels 压缩回 channels
        self.fusion = nn.Sequential(
            Conv1dSame(self.cat_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        # ---------- 注意力统计池化 ----------
        self.pooling = AttentiveStatisticsPooling(channels, attention_channels=128)

        # ---------- 全连接投影 ----------
        # pooling 输出为 2*channels (均值+标准差) → embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(channels * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """对卷积和全连接层进行权重初始化。

        Conv1d → Kaiming (He) 正态初始化，适配 ReLU 非线性
        Linear → 正交初始化，保持嵌入空间的各向同性
        """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _compute_fbank_lengths(
        self,
        waveform_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """根据波形原始采样点数，精确估算经过 MelSpectrogram 后的帧数。

        帧数公式: (n_samples - n_fft) // hop_length + 1

        对于短于 n_fft 的极短音频，返回至少 1 帧。

        Parameters
        ----------
        waveform_lengths : torch.Tensor
            波形原始采样点数（不含 padding），shape [B]。

        Returns
        -------
        torch.Tensor
            对应的 Fbank 特征有效帧数，shape [B]。
        """
        hop_length = 160   # 与 MelSpectrogram 的 hop_length 一致
        n_fft = 512        # 与 MelSpectrogram 的 n_fft 一致
        lengths = ((waveform_lengths - n_fft) // hop_length + 1).clamp(min=1)
        return lengths.int()

    def forward(
        self,
        enroll_wavs: torch.Tensor,
        enroll_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播：从原始波形提取声纹嵌入。

        完整流程:
            波形 → MelSpectrogram → Log → Conv1D 前端 → 3×SE-Res2Block
            → 多层级联 → 1×1 融合 → ASP (mask padding) → Linear+BN → L2 Norm

        Parameters
        ----------
        enroll_wavs : torch.Tensor
            唤醒音频批次，shape [B, 1, T_wav]。
            由于批次内音频长度不同，末尾部分为 padding 零值。
        enroll_lengths : torch.Tensor
            各样本的真实采样点数（不含 padding），shape [B]。
            用于推导 Fbank 帧级有效长度，在 ASP 池化时生成 mask，
            确保 padding 静音帧不参与声纹统计。

        Returns
        -------
        torch.Tensor
            L2 归一化的说话人嵌入向量，shape [B, embedding_dim]。
            可直接用于余弦相似度计算或作为下游 TSE/ASR 的条件输入。
        """
        # ---------- Step 1: MelSpectrogram ----------
        # MelSpectrogram 期望输入 shape [B, T]（单声道）
        if enroll_wavs.dim() == 3:
            wav = enroll_wavs.squeeze(1)  # [B, 1, T] → [B, T]
        else:
            wav = enroll_wavs

        # 提取 Mel 功率谱
        mel = self.mel_spec(wav)  # [B, n_mels, T_feat]

        # Log 压缩，模拟人耳对数响度感知，防止 log(0) 加小常数
        mel = torch.log(mel + 1e-6)

        # ---------- Step 2: Conv1D 前端 ----------
        feat = self.frontend(mel)  # [B, channels, T_feat]

        # ---------- Step 3: 计算 Fbank 帧级有效长度 ----------
        fbank_lengths = self._compute_fbank_lengths(enroll_lengths)
        # clamp 防止浮点/整数误差导致的帧数略超实际
        fbank_lengths = torch.clamp(fbank_lengths, max=feat.shape[2])

        # ---------- Step 4: SE-Res2Block 堆叠 ----------
        # 三层 SE-Res2Block，逐步提取多尺度声纹特征
        out1 = self.block1(feat)  # [B, channels, T]
        out2 = self.block2(out1)  # [B, channels, T]
        out3 = self.block3(out2)  # [B, channels, T]

        # ---------- Step 5: 多层特征拼接与融合 ----------
        # 拼接浅中深三层输出 → 1×1 卷积融合
        multi_scale = torch.cat([out1, out2, out3], dim=1)  # [B, 3*channels, T]
        fused = self.fusion(multi_scale)                     # [B, channels, T]

        # ---------- Step 6: 注意力统计池化 ----------
        # 结合 fbank_lengths 生成 mask，padding 帧权重→0
        pooled = self.pooling(fused, lengths=fbank_lengths)  # [B, 2*channels]

        # ---------- Step 7: 输出投影 ----------
        embedding = self.embedding_layer(pooled)  # [B, embedding_dim]

        # L2 归一化，使得 ||embedding||₂ = 1
        # 归一化后可直接用点积/余弦相似度衡量说话人相似性
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def get_num_params(self) -> Tuple[int, int]:
        """统计参数量。

        Returns
        -------
        Tuple[int, int]
            (总参数量, 可训练参数量)。
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ====================================================================
# 主程序入口（快速自检）
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LightSpeakerEncoder 轻量级 ECAPA-TDNN 自检")
    print("=" * 60)

    # ---------- 1. 构造模型 ----------
    model = LightSpeakerEncoder(
        embedding_dim=256,
        n_mels=80,
        channels=256,         # 轻量化通道数
        sample_rate=16000,
    )

    total_params, trainable_params = model.get_num_params()
    print(
        f"\n[参数量统计] "
        f"total={total_params:,}, "
        f"trainable={trainable_params:,}"
    )

    # ---------- 2. 模拟输入 ----------
    batch_size = 2
    # 模拟两条不同长度的音频: 2.0s 和 1.0s @ 16kHz
    audio_lengths = torch.tensor([32000, 16000], dtype=torch.long)
    max_length = audio_lengths.max().item()

    # 随机波形，padding 部分填 0（模拟 DataLoader 的 collate_fn 行为）
    dummy_wavs = torch.zeros(batch_size, 1, max_length, dtype=torch.float32)
    for i, length in enumerate(audio_lengths):
        dummy_wavs[i, 0, :length] = torch.randn(length) * 0.01

    print(f"\n[输入形状] {dummy_wavs.shape}  (B={batch_size}, 1, max_T={max_length})")
    print(f"[音频长度] {audio_lengths.tolist()}  (2.0s, 1.0s @ 16kHz)")

    # ---------- 3. 前向传播 ----------
    model.eval()
    with torch.inference_mode():
        embeddings = model(dummy_wavs, audio_lengths)

    print(f"\n[输出形状] {embeddings.shape}  (B={batch_size}, embedding_dim=256)")
    print(f"[L2 范数]   {embeddings.norm(p=2, dim=1).tolist()}  # 应全为 1.0")

    # ---------- 4. 结构概览 ----------
    print(f"\n[网络结构]")
    print(f"  - MelSpectrogram: 80-dim Fbank (25ms win, 10ms hop)")
    print(f"  - Conv1D Frontend: 80 → {model.channels} (1×1 conv)")
    print(f"  - SE-Res2Block ×3: channels={model.channels}, scale=4")
    print(f"  - Multi-layer Cat: {model.channels}×3 → 1×1 fusion → {model.channels}")
    print(f"  - ASP Pooling: {model.channels} → {model.channels*2} (± mask)")
    print(f"  - Embedding: {model.channels*2} → 256 + L2 Norm")

    print("\n自检完成！模型总参数量 = {:,}".format(total_params))