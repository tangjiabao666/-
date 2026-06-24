#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSE-ASR 项目 —— 轻量级目标说话人提取网络 (LightTSExtractor)。

本模块实现了一个基于 Conv-TasNet 架构思想的时域掩码网络，
通过将目标说话人的声纹嵌入 (Speaker Embedding) 注入分离器，
从混合音频中提取出仅含目标说话人的干净波形。

网络结构概览
------------
::

    mixed_wavs [B, 1, T_in]
           │
           ▼
    ┌─────────────────────┐
    │  Encoder (Conv1d)   │  → 将波形编码为帧级特征 [B, N_feat, T_enc]
    │  kernel=16, stride=8│     (8 倍下采样，减少 TCN 计算量)
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  Speaker FiLM       │  → 用 speaker_embedding [B, emb_dim] 调制特征
    │  (Feature-wise      │     γ, β = Linear(emb → N_feat)
    │   Linear Modulation)│     特征 = γ * 特征 + β
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────────────┐
    │  TCN Separator (重复 2 次)   │
    │  ┌─────────────────────┐    │
    │  │  DConv Block ×5     │    │  dilation = 1, 2, 4, 8, 16
    │  │  (Depthwise Sep)    │    │  每个 block: 1×1 expand → DW Conv →
    │  └─────────────────────┘    │   1×1 project + residual + PReLU
    │  ... repeat 2 ...           │
    └─────────────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  Mask (Sigmoid)     │  → 输出范围 [0, 1] 的软掩码 [B, N_feat, T_enc]
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  Decoder            │  → ConvTranspose1d 将特征还原为波形
    │  (ConvTranspose1d)  │     [B, N_feat, T_enc] → [B, 1, T_out]
    │  kernel=16, stride=8│     裁剪/填充至与输入严格等长
    └─────────────────────┘
           │
           ▼
    extracted_wavs [B, 1, T_in]

参考论文
--------
- Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for
  Speech Separation (Luo & Mesgarani, TASLP 2019)
- FiLM: Feature-wise Linear Modulation (Perez et al., AAAI 2018)
- SpEx+: A Complete Time Domain Speaker Extraction Network
  (Ge et al., Interspeech 2020)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# 基础组件 1：深度可分离残差卷积块 (Depthwise Separable Conv Block)
# ====================================================================

class DConvBlock(nn.Module):
    """深度可分离残差卷积块——TCN 的基本组成单元。

    每个 Block 的结构为::

        Input [B, C_in, T]
            │
            ├──→ 1×1 Conv (C_in → C_hidden)   # 通道扩展
            │    + PReLU
            │    + LayerNorm (通道维)
            │
            ├──→ DW Conv1d (C_hidden, groups=C_hidden, dilation=d)
            │    # 深度可分离卷积，每个通道独立做膨胀卷积
            │    # 膨胀率递增增大感受野
            │    + PReLU
            │    + LayerNorm (通道维)
            │
            ├──→ 1×1 Conv (C_hidden → C_in)   # 通道压缩回原始维度
            │    + Residual Add
            │
            └──→ Output [B, C_in, T]

    深度可分离卷积 (Depthwise Separable Conv) 将标准卷积分解为:
        1. Depthwise: 每个通道独立卷积 (groups = C_hidden)
        2. Pointwise: 1×1 跨通道融合
    这大幅减少了参数量和计算量，非常适合轻量化设计。

    Parameters
    ----------
    channels : int
        输入/输出通道数 (C_in = C_out)。
    hidden_channels : int
        中间扩展通道数 (C_hidden)。通常设为 2×channels 以获得足够的容量。
    kernel_size : int
        深度可分离卷积的核大小，默认 3。
    dilation : int
        膨胀率。膨胀率递增序列 (如 1,2,4,8,16) 使网络逐层扩大感受野，
        从而捕捉不同时间尺度的上下文信息。
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        # 计算 same padding，确保时间维度不变
        padding = (kernel_size - 1) * dilation // 2

        # ---------- 1×1 通道扩展 ----------
        self.expand_conv = nn.Conv1d(channels, hidden_channels, kernel_size=1)
        self.expand_norm = nn.LayerNorm(hidden_channels)
        self.expand_act = nn.PReLU(hidden_channels)

        # ---------- 深度可分离卷积 (Depthwise Conv) ----------
        self.dw_conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=hidden_channels,       # depthwise: 每个通道独立卷积
        )
        self.dw_norm = nn.LayerNorm(hidden_channels)
        self.dw_act = nn.PReLU(hidden_channels)

        # ---------- 1×1 通道压缩 ----------
        self.project_conv = nn.Conv1d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入特征，shape [B, C, T]。

        Returns
        -------
        torch.Tensor
            残差连接后的输出，shape [B, C, T]。
        """
        residual = x

        # 通道扩展
        out = self.expand_conv(x)                       # [B, C→H, T]
        out = out.permute(0, 2, 1)                      # [B, T, H]
        out = self.expand_norm(out).permute(0, 2, 1)   # [B, H, T]
        out = self.expand_act(out)

        # 深度可分离卷积
        out = self.dw_conv(out)                         # [B, H, T]
        out = out.permute(0, 2, 1)                      # [B, T, H]
        out = self.dw_norm(out).permute(0, 2, 1)       # [B, H, T]
        out = self.dw_act(out)

        # 通道压缩
        out = self.project_conv(out)                    # [B, H→C, T]

        # 残差连接
        out = out + residual

        return out


# ====================================================================
# 基础组件 2：TCN 堆叠 (Temporal Convolutional Network Stack)
# ====================================================================

class TCNStack(nn.Module):
    """TCN 堆叠模块——由多个重复组 (Repeat) 组成。

    每个 Repeat 包含 K 个 DConvBlock，膨胀率依次为 [1, 2, 4, 8, ..., 2^(K-1)]。
    Repeat 之间共享相同的膨胀率模式，使网络在多个分辨率级别上反复处理特征。

    轻量化设计：
    - repeats=2（而非原始 Conv-TasNet 的 3 次）
    - blocks_per_repeat=5，dilations = [1, 2, 4, 8, 16]
    - 深度可分离卷积替代标准卷积

    Parameters
    ----------
    channels : int
        输入/输出通道数。
    hidden_channels : int
        DConvBlock 中间层扩展通道数。
    kernel_size : int, optional
        DConvBlock 核大小，默认 3。
    repeats : int, optional
        TCN 重复次数，默认 2。
    blocks_per_repeat : int, optional
        每个 Repeat 内的 block 数，默认 5（dilations=[1,2,4,8,16]）。
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        repeats: int = 2,
        blocks_per_repeat: int = 5,
    ) -> None:
        super().__init__()

        # 膨胀率序列：1, 2, 4, 8, 16, ..., 2^(K-1)
        dilations = [2 ** i for i in range(blocks_per_repeat)]

        # 构建所有 DConvBlock
        self.blocks = nn.ModuleList()
        for _ in range(repeats):
            for d in dilations:
                self.blocks.append(
                    DConvBlock(
                        channels=channels,
                        hidden_channels=hidden_channels,
                        kernel_size=kernel_size,
                        dilation=d,
                    )
                )

        self.num_blocks = len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，逐 block 串行。

        Parameters
        ----------
        x : torch.Tensor
            输入特征，shape [B, C, T]。

        Returns
        -------
        torch.Tensor
            TCN 处理后的特征，shape [B, C, T]。
        """
        for block in self.blocks:
            x = block(x)
        return x


# ====================================================================
# 主网络：LightTSExtractor
# ====================================================================

class LightTSExtractor(nn.Module):
    """轻量级目标说话人提取网络。

    基于 Conv-TasNet 框架思想，通过 FiLM 机制将目标说话人的声纹嵌入
    注入 TCN 分离器，从混合音频中提取仅含目标说话人的干净语音。

    设计要点:
    1. Encoder 进行 8 倍下采样，大幅降低 TCN 计算开销
    2. FiLM 逐通道调制，将声纹信息作为条件注入特征
    3. TCN 使用深度可分离卷积 + 膨胀因果卷积，兼顾感受野与轻量性
    4. Decoder 使用转置卷积还原波形，并精确裁剪/填充保证长度一致
    5. Sigmoid Mask 保证输出掩码在 [0,1] 范围内

    Parameters
    ----------
    in_channels : int, optional
        输入音频通道数，默认 1（单声道）。
    out_channels : int, optional
        输出音频通道数，默认 1。
    feature_dim : int, optional
        Encoder 输出 / TCN 处理的通道维度，默认 256。
    hidden_dim : int, optional
        DConvBlock 中间扩展通道数，默认 512。
    emb_dim : int, optional
        说话人嵌入向量维度，与 LightSpeakerEncoder 输出对齐，默认 256。
    kernel_size : int, optional
        Encoder/Decoder 卷积核大小，默认 16。
    stride : int, optional
        Encoder/Decoder 步长（下采样比率），默认 8。
    tcn_kernel : int, optional
        TCN 内部 DW Conv 的核大小，默认 3。
    repeats : int, optional
        TCN 重复次数，默认 2。
    blocks_per_repeat : int, optional
        每个 Repeat 的 DConvBlock 数量，默认 5。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        emb_dim: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
        tcn_kernel: int = 3,
        repeats: int = 2,
        blocks_per_repeat: int = 5,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        self.stride = stride
        # Encoder 两侧 padding 量，使用 'same' 风格 padding
        self.enc_padding = kernel_size // 2

        # ============================================================
        # Encoder: 1D Conv 将波形编码为帧级特征
        # ============================================================
        # 输入 [B, 1, T_in] → 输出 [B, feature_dim, T_enc]
        # T_enc ≈ (T_in + 2*enc_padding - kernel_size) // stride + 1
        self.encoder = nn.Conv1d(
            in_channels=in_channels,
            out_channels=feature_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.enc_padding,
        )

        # ============================================================
        # Speaker FiLM: 特征级线性调制
        # ============================================================
        # FiLM 机制: 给定条件向量 c (speaker_embedding)，生成:
        #   γ = Linear(c)  — 缩放因子
        #   β = Linear(c)  — 偏移因子
        # 调制特征: modulated = γ ⊙ feature + β
        #
        # 这种逐通道调制比简单的拼接更灵活，每个通道可根据声纹信息
        # 独立调整激活强度，更好地实现"选择性提取目标说话人"。
        self.film_gamma = nn.Linear(emb_dim, feature_dim)
        self.film_beta = nn.Linear(emb_dim, feature_dim)

        # ============================================================
        # TCN Separator: 时序卷积分离器
        # ============================================================
        self.tcn = TCNStack(
            channels=feature_dim,
            hidden_channels=hidden_dim,
            kernel_size=tcn_kernel,
            repeats=repeats,
            blocks_per_repeat=blocks_per_repeat,
        )

        # ============================================================
        # Mask 分支 + Feature 分支
        # ============================================================
        # 原始 Conv-TasNet 使用两个独立分支:
        #   1. Mask 分支: 输出 [0,1] 范围的软掩码
        #   2. Encoder 输出作为特征 (也可再加处理)
        # 这里采用简化方案: TCN 输出经过 Sigmoid 产生 mask，
        # 再与 Encoder 原始输出相乘
        self.mask_proj = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid(),    # 确保 mask ∈ [0, 1]
        )

        # ============================================================
        # Decoder: ConvTranspose1d 将特征还原为波形
        # ============================================================
        # 输入 [B, feature_dim, T_enc] → 输出 [B, out_channels, T_out_raw]
        self.decoder = nn.ConvTranspose1d(
            in_channels=feature_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.enc_padding,
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """对卷积和线性层进行权重初始化。"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        mixed_wavs: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播：从混合音频中提取目标说话人的干净波形。

        完整流程:
            mixed_wavs → Encoder → FiLM(声纹调制) → TCN → Mask ⊙ Encoder_out
            → Decoder (ConvTranspose1d) → 长度裁剪与输入对齐

        Parameters
        ----------
        mixed_wavs : torch.Tensor
            混合音频波形，shape [B, in_channels, T_in]。
            通常为带有噪声、混响或多说话人干扰的音频。
        speaker_embedding : torch.Tensor
            目标说话人的声纹嵌入向量，shape [B, emb_dim]。
            由 LightSpeakerEncoder 提取并传入。

        Returns
        -------
        torch.Tensor
            提取后的干净波形，shape [B, out_channels, T_in]。
            输出长度严格与输入长度一致。
        """
        # 记录输入长度，用于输出对齐
        input_length = mixed_wavs.shape[-1]

        # ---------- Step 1: Encoder ----------
        # 波形 → 帧级特征 [B, feature_dim, T_enc]
        # T_enc ≈ ceil((T_in + 2*enc_padding - kernel_size) / stride) + 1
        enc_out = self.encoder(mixed_wavs)              # [B, feature_dim, T_enc]

        # 保存一份 Encoder 原始输出，后续与 mask 相乘
        enc_feat = enc_out                              # [B, feature_dim, T_enc]

        # ---------- Step 2: FiLM 声纹调制 ----------
        # 从 speaker_embedding 生成逐通道的 γ (缩放) 和 β (偏移)
        # gamma, beta: [B, feature_dim] → [B, feature_dim, 1]
        gamma = self.film_gamma(speaker_embedding).unsqueeze(-1)  # [B, F, 1]
        beta = self.film_beta(speaker_embedding).unsqueeze(-1)    # [B, F, 1]

        # 调制: modulated_feat = γ ⊙ feat + β
        # 这相当于告诉网络"我们要提取这个人的声音"
        modulated = gamma * enc_out + beta              # [B, feature_dim, T_enc]

        # ---------- Step 3: TCN 处理 ----------
        # 多层膨胀卷积提取时序上下文
        tcn_out = self.tcn(modulated)                   # [B, feature_dim, T_enc]

        # ---------- Step 4: 计算 Mask ----------
        # TCN 输出 → 1×1 conv → ReLU → 1×1 conv → Sigmoid → mask ∈ [0,1]
        mask = self.mask_proj(tcn_out)                  # [B, feature_dim, T_enc]

        # ---------- Step 5: 掩码应用 ----------
        # 将 mask 与 Encoder 原始特征逐元素相乘
        # 只保留属于目标说话人的时频成分
        masked_feat = enc_feat * mask                   # [B, feature_dim, T_enc]

        # ---------- Step 6: Decoder ----------
        # 转置卷积将帧级特征还原回时域波形
        reconstructed = self.decoder(masked_feat)       # [B, out_channels, T_out_raw]

        # ---------- Step 7: 长度对齐 ----------
        # ConvTranspose1d 的输出长度可能与输入不完全一致
        # 裁剪或填充至与输入严格等长
        out_length = reconstructed.shape[-1]
        if out_length > input_length:
            # 输出过长 → 裁剪右侧
            reconstructed = reconstructed[..., :input_length]
        elif out_length < input_length:
            # 输出过短 → 右侧零填充
            pad_amount = input_length - out_length
            reconstructed = F.pad(reconstructed, (0, pad_amount))

        return reconstructed                                     # [B, out_channels, T_in]

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
    print("LightTSExtractor 轻量级 TSE 网络自检")
    print("=" * 60)

    # ---------- 1. 构造模型 ----------
    model = LightTSExtractor(
        in_channels=1,
        out_channels=1,
        feature_dim=256,         # TCN 通道数
        hidden_dim=512,          # DConvBlock 内部扩展通道
        emb_dim=256,             # 声纹嵌入维度（与 LightSpeakerEncoder 对齐）
        kernel_size=16,
        stride=8,
        tcn_kernel=3,
        repeats=2,               # 轻量化：仅 2 次 TCN 重复
        blocks_per_repeat=5,     # dilations=[1,2,4,8,16]
    )

    total_params, trainable_params = model.get_num_params()
    print(f"\n[参数量统计] total={total_params:,}, trainable={trainable_params:,}")

    # ---------- 2. 模拟输入 ----------
    batch_size = 2
    audio_length = 32000        # 2.0 秒 @ 16kHz

    # 模拟混合音频（带 noise）
    dummy_mixed = torch.randn(batch_size, 1, audio_length) * 0.5

    # 模拟说话人嵌入（来自 LightSpeakerEncoder）
    dummy_spk_emb = F.normalize(torch.randn(batch_size, 256), p=2, dim=1)

    print(f"\n[输入形状]")
    print(f"  mixed_wavs      : {dummy_mixed.shape}")
    print(f"  speaker_embed   : {dummy_spk_emb.shape}")

    # ---------- 3. 前向传播 ----------
    model.eval()
    with torch.inference_mode():
        extracted = model(dummy_mixed, dummy_spk_emb)

    print(f"\n[输出形状]")
    print(f"  extracted_wavs  : {extracted.shape}")
    print(f"  长度一致性检查  : input={audio_length}, output={extracted.shape[-1]}")
    assert extracted.shape == dummy_mixed.shape, (
        f"输出形状不匹配！期望 {dummy_mixed.shape}，实际 {extracted.shape}"
    )

    # ---------- 4. 结构统计 ----------
    print(f"\n[网络结构]")
    print(f"  - Encoder (Conv1d):       1 → 256, kernel={model.kernel_size}, stride={model.stride}")
    print(f"  - FiLM (Linear×2):       256 → 256 (γ + β)")
    print(f"  - TCN Stack:             2 repeats × 5 blocks = 10 DConvBlocks")
    print(f"    - 每 block: 1×1(256→512) → DW Conv(k=3,d=dil) → 1×1(512→256)")
    print(f"    - Dilations:            [1,2,4,8,16] × 2 repeats")
    print(f"  - Mask Proj:             256 → 256 → 256 + Sigmoid")
    print(f"  - Decoder (ConvTrans1d): 256 → 1, kernel=16, stride=8")
    print(f"  - 长度对齐:               裁剪/填充至与输入等长")
    print(f"\n  总参数量: {total_params:,}")

    print("\n自检完成！")