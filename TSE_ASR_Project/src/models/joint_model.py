#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSE-ASR 项目 —— 端到端联合大模型 (JointTSEASR)。

本模块将已构建的三个子模块——声纹提取器 (LightSpeakerEncoder)、
目标说话人提取器 (LightTSExtractor)、轻量级 ASR 后端 (LightASRBackend)
以及拒识分类头 (RejectionHead)——整合到一个统一的 ``JointTSEASR`` 中，
实现从原始多轮交互音频到识别文本 + 拒识判断的端到端推理。

核心调度流程
------------
::

                     enroll_wavs [B, 1, T_enr]
                          │
                          ▼
               ┌─────────────────────┐
               │  LightSpeakerEncoder│  → speaker_embed [B, 256]
               └─────────────────────┘
                          │
                          ▼
    mixed_wavs [B, 1, T_mix] ──┬──→ speaker_embed
                               │
                               ▼
               ┌─────────────────────┐
               │  LightTSExtractor   │  → clean_wavs [B, 1, T_mix]
               └─────────────────────┘
                    │               │
                    ▼               ▼
       ┌──────────────────┐  ┌──────────────────┐
       │  LightASRBackend │  │  RejectionHead   │
       │  (Conformer+CTC) │  │  (MelPool+MLP)   │
       └──────────────────┘  └──────────────────┘
                    │               │
                    ▼               ▼
           asr_logits              reject_logits
        [B, T_enc, V]              [B, 2]

返回字典:
    {
        "clean_wavs":    ...,
        "reject_logits": ...,
        "asr_logits":    ...,
    }
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# 导入已有子模块
# 兼容两种运行方式：作为包导入（相对路径）或直接运行脚本（绝对路径）
try:
    from .speaker_encoder.encoder import LightSpeakerEncoder
    from .tse_network.extractor import LightTSExtractor
except ImportError:
    # 直接运行脚本时的绝对导入（需 PYTHONPATH 指向 src/models/ 或项目根目录）
    import sys
    import os as _os
    _current_dir = _os.path.dirname(_os.path.abspath(__file__))
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)
    from speaker_encoder.encoder import LightSpeakerEncoder
    from tse_network.extractor import LightTSExtractor


# ====================================================================
# Conv2D 降采样模块 (Conformer 标准前端)
# ====================================================================

class Conv2DSubsampling(nn.Module):
    """Conformer 标准前端：2 层 Conv2D 实现 4 倍时间维度降采样。

    将 Mel 频谱特征 (T × dim) 通过两层 stride=2 的 Conv2D 进行降采样，
    然后通过线性层将通道映射到 Conformer 的输入维度。

    具体流程::

        输入 [B, T, dim]
            │  unsqueeze(1)
            ▼
        [B, 1, T, dim]
            │  Conv2d(1→32, k=3, stride=2)  → CELU
            ▼
        [B, 32, T/2, dim/2]
            │  Conv2d(32→64, k=3, stride=2) → CELU
            ▼
        [B, 64, T/4, dim/4]
            │  reshape → [B, T/4, 64*dim/4]
            │  Linear → [B, T/4, out_dim]
            ▼
        输出 [B, T/4, out_dim]

    输出长度的计算公式:
        T_out = ((T_in - 1) // 2 - 1) // 2 ≈ T_in // 4

    Parameters
    ----------
    input_dim : int
        输入特征维度（通常为 Fbank 的 n_mels=80）。
    output_dim : int
        输出特征维度（通常与 Conformer 的 input_dim 对齐，如 256）。
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        # 第一层 Conv2D: stride=(2, 2)，在时间和频率两个方向降采样
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        # 计算经过两层 stride=2 后的特征维度
        # 输入: [B, 1, T, input_dim]
        # 经过 conv1: T' = (T - 3) // 2 + 1 ≈ T//2, D' = (D - 3) // 2 + 1
        # 经过 conv2: T'' ≈ T/4, D'' = (D' - 3) // 2 + 1
        # 简化：直接估算为 T/4 和 D/4
        # 精确值难以在 __init__ 中确定（依赖输入长度），使用 Linear 动态适应
        self.output_dim = output_dim
        self.linear = nn.Linear(64 * ((input_dim - 3) // 2 + 1 - 3) // 2 + 64, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入特征，shape [B, T, input_dim]。
        lengths : torch.Tensor, optional
            各样本原始帧数，shape [B]。用于计算降采样后的有效长度。

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            - 降采样后的特征 [B, T_out, output_dim]
            - 降采样后的长度 [B]（如果传入了 lengths）
        """
        # [B, T, C] → [B, 1, T, C]
        x = x.unsqueeze(1)

        # 两层 Conv2D + CELU 激活
        x = F.celu(self.conv1(x), alpha=1.2)       # [B, 32, T/2, ~C/2]
        x = F.celu(self.conv2(x), alpha=1.2)       # [B, 64, T/4, ~C/4]

        # 展平通道维度： [B, 64, T', C'] → [B, T', 64*C']
        batch_size, channels, time_dim, freq_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()     # [B, T', 64, C']
        x = x.view(batch_size, time_dim, channels * freq_dim)  # [B, T', 64*C']

        # 线性投影到输出维度
        x = self.linear(x)                          # [B, T', output_dim]

        # 更新长度
        out_lengths: Optional[torch.Tensor] = None
        if lengths is not None:
            # 每层 stride=2 的降采样公式
            lengths1 = (lengths - 3) // 2 + 1
            out_lengths = (lengths1 - 3) // 2 + 1
            out_lengths = out_lengths.clamp(min=1)

        return x, out_lengths


# ====================================================================
# 轻量级 ASR 后端 (LightASRBackend)
# ====================================================================

class LightASRBackend(nn.Module):
    """轻量级 ASR 后端——Conformer 骨干 + CTC 输出层。

    使用 torchaudio.models.Conformer 作为核心序列建模网络，
    前端通过 Conv2D 降采样将 Fbank 特征压缩 4 倍后送入 Conformer。

    Parameters
    ----------
    input_dim : int, optional
        输入 Fbank 特征维度，默认 80。
    conformer_dim : int, optional
        Conformer 内部维度，默认 256。
    conformer_layers : int, optional
        Conformer 层数，默认 4（轻量化设计）。
    conformer_heads : int, optional
        多头注意力头数，默认 4。
    ffn_dim : int, optional
        Conformer FFN 中间维度，默认 1024。
    vocab_size : int, optional
        输出字表大小（含 blank），默认 4000。
    dropout : float, optional
        Conformer 内部 dropout 率，默认 0.1。
    """

    def __init__(
        self,
        input_dim: int = 80,
        conformer_dim: int = 256,
        conformer_layers: int = 4,
        conformer_heads: int = 4,
        ffn_dim: int = 1024,
        vocab_size: int = 4000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.conformer_dim = conformer_dim

        # ---------- Fbank 提取 ----------
        # 25ms 窗口，10ms 帧移
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            n_mels=input_dim,
            power=2.0,
        )

        # ---------- Conv2D 降采样前端 ----------
        # 将 80-dim Mel 降采样为 conformer_dim 维特征
        self.subsampling = Conv2DSubsampling(input_dim, conformer_dim)

        # ---------- Conformer 骨干 ----------
        self.conformer = torchaudio.models.Conformer(
            input_dim=conformer_dim,
            num_heads=conformer_heads,
            ffn_dim=ffn_dim,
            num_layers=conformer_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout,
        )

        # ---------- CTC 输出层 ----------
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(conformer_dim),
            nn.Linear(conformer_dim, vocab_size),
        )
        # 注意：CTC 需要 log_softmax，在 loss 函数中处理

        self.vocab_size = vocab_size

        # 权重初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化线性层的权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _compute_mel_lengths(self, waveform_lengths: torch.Tensor) -> torch.Tensor:
        """根据波形采样点数计算 Mel 频谱的帧数。

        Parameters
        ----------
        waveform_lengths : torch.Tensor
            波形采样点数，shape [B]。

        Returns
        -------
        torch.Tensor
            Mel 频谱帧数，shape [B]。
        """
        n_fft = 512
        hop_length = 160
        lengths = ((waveform_lengths - n_fft) // hop_length + 1).clamp(min=1)
        return lengths.int()

    def forward(
        self,
        clean_wavs: torch.Tensor,
        mixed_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播：从提纯后波形计算 ASR 字符级 logits。

        Parameters
        ----------
        clean_wavs : torch.Tensor
            TSE 提取后的干净波形，shape [B, 1, T_wav]。
        mixed_lengths : torch.Tensor, optional
            各样本原始采样点数，shape [B]。用于计算 CTC 所需的长度。

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            - asr_logits: [B, T_enc, vocab_size]（CTC 输出，不含 softmax）
            - asr_lengths: [B]（编码后的帧数，用于 CTC loss 的 input_lengths）
        """
        # ---------- Step 1: MelSpectrogram ----------
        if clean_wavs.dim() == 3:
            wav = clean_wavs.squeeze(1)  # [B, T]
        else:
            wav = clean_wavs

        mel = self.mel_spec(wav)          # [B, input_dim, T_mel]
        mel = torch.log(mel + 1e-6)       # log 压缩

        # ---------- Step 2: 转换为 [B, T, C] 格式 ----------
        mel = mel.permute(0, 2, 1)        # [B, T_mel, input_dim]

        # ---------- Step 3: Conv2D 降采样 ----------
        # 80-dim Mel → 256-dim 特征，4 倍时间压缩
        feat, _ = self.subsampling(mel, None)

        # ---------- Step 4: Conformer 编码 ----------
        # 计算降采样后的有效帧数，并与 feat 实际时间维度对齐
        if mixed_lengths is not None:
            enc_lengths = self._compute_mel_lengths(mixed_lengths)
            # 4 倍降采样: ((T - 1) // 2 - 1) // 2 + 1 ≈ ceil(T / 4)
            enc_lengths = torch.ceil(enc_lengths.float() / 4.0).long()
            enc_lengths = enc_lengths.clamp(min=1, max=feat.shape[1])
            max_valid_len = int(enc_lengths.max().item())
            if max_valid_len < feat.shape[1]:
                feat = feat[:, :max_valid_len, :]
            enc_lengths = enc_lengths.clamp(min=1, max=feat.shape[1])
        else:
            enc_lengths = None
        conformer_out, output_lengths = self.conformer(feat, enc_lengths)
        # conformer_out: [B, T_enc, conformer_dim]

        # ---------- Step 6: CTC 输出投影 ----------
        logits = self.ctc_head(conformer_out)  # [B, T_enc, vocab_size]

        return logits, output_lengths


# ====================================================================
# 拒识分类头 (Rejection Head)
# ====================================================================

class WaveformFusionGate(nn.Module):
    """Blend TSE output with the original mixture to avoid over-suppression."""

    def __init__(self, emb_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        # Start close to the strong mixed-audio baseline, then learn when to trust TSE.
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(
        self,
        mixed_wavs: torch.Tensor,
        tse_wavs: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed_abs = mixed_wavs.abs().mean(dim=(1, 2)).unsqueeze(1)
        tse_abs = tse_wavs.abs().mean(dim=(1, 2)).unsqueeze(1)
        diff_abs = (mixed_wavs - tse_wavs).abs().mean(dim=(1, 2)).unsqueeze(1)
        rel_diff = diff_abs / (mixed_abs + 1e-6)
        stats = torch.cat([mixed_abs, tse_abs, diff_abs, rel_diff], dim=1)
        gate = torch.sigmoid(self.net(torch.cat([speaker_embedding, stats], dim=1)))
        gate = gate.view(-1, 1, 1)
        fused_wavs = gate * tse_wavs + (1.0 - gate) * mixed_wavs
        return fused_wavs, gate


class RejectionHead(nn.Module):
    """拒识分类头——判断提纯音频中是否包含目标说话人。

    原理:
        1. 从 clean_wavs 中提取 Mel 频谱并做全局时间池化，得到声学统计量
        2. 将声学统计量与 speaker_embedding 拼接
        3. 通过浅层 MLP 输出二分类 logits [B, 2]
           第 0 类: 拒识 (neg) — 提纯音频中不包含目标说话人
           第 1 类: 正样本 (pos) — 提纯音频中包含目标说话人

    Parameters
    ----------
    n_mels : int, optional
        Mel 滤波器组数量，默认 80。
    emb_dim : int, optional
        说话人嵌入维度，默认 256。
    hidden_dim : int, optional
        MLP 隐藏层维度，默认 256。
    """

    def __init__(
        self,
        n_mels: int = 80,
        emb_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # ---------- Mel 频谱提取 ----------
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            n_mels=n_mels,
            power=2.0,
        )

        # ---------- 分类器 ----------
        # 输入: 声学特征池化向量 [B, n_mels] + 声纹嵌入 [B, emb_dim]
        #      → [B, n_mels + emb_dim]
        # 输出: [B, 2]（二分类 logits）
        self.contrast_proj = nn.Sequential(
            nn.Linear(n_mels * 3, n_mels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_mels, n_mels),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_mels + emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2),   # 2 类: neg=0, pos=1
        )

        # 权重初始化
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for module in self.contrast_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.constant_(self.contrast_proj[-1].weight, 0)
        nn.init.constant_(self.contrast_proj[-1].bias, 0)

    def forward(
        self,
        clean_wavs: torch.Tensor,
        speaker_embedding: torch.Tensor,
        mixed_wavs: Optional[torch.Tensor] = None,
        extracted_wavs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播：计算拒识二分类 logits。

        Parameters
        ----------
        clean_wavs : torch.Tensor
            TSE 提取后的干净波形，shape [B, 1, T_wav]。
        speaker_embedding : torch.Tensor
            目标说话人声纹嵌入，shape [B, emb_dim]。

        Returns
        -------
        torch.Tensor
            二分类 logits，shape [B, 2]。
            reject_logits[:, 0] → 拒识 (neg) 得分
            reject_logits[:, 1] → 正样本 (pos) 得分
        """
        # ---------- Step 1: Mel 频谱 ----------
        if clean_wavs.dim() == 3:
            wav = clean_wavs.squeeze(1)  # [B, T]
        else:
            wav = clean_wavs

        mel = self.mel_spec(wav)          # [B, n_mels, T_mel]
        mel = torch.log(mel + 1e-6)

        # ---------- Step 2: 全局时间池化 ----------
        # 对整个音频的时间维度做平均，得到一个固定长度的声学表征
        acoustic_feat = mel.mean(dim=2)   # [B, n_mels]

        if mixed_wavs is not None:
            mixed_feat = self._pool_audio(mixed_wavs)
            diff_feat = (mixed_feat - acoustic_feat).abs()
            if extracted_wavs is not None:
                extracted_feat = self._pool_audio(extracted_wavs)
            else:
                extracted_feat = acoustic_feat
            contrast_feat = torch.cat([mixed_feat, extracted_feat, diff_feat], dim=1)
            acoustic_feat = acoustic_feat + self.contrast_proj(contrast_feat)

        # ---------- Step 3: 拼接声学特征与声纹嵌入 ----------
        combined = torch.cat([acoustic_feat, speaker_embedding], dim=1)
        # [B, n_mels + emb_dim]

        # ---------- Step 4: MLP 分类 ----------
        logits = self.classifier(combined)  # [B, 2]

        return logits

    def _pool_audio(self, wavs: torch.Tensor) -> torch.Tensor:
        if wavs.dim() == 3:
            wav = wavs.squeeze(1)
        else:
            wav = wavs
        mel = self.mel_spec(wav)
        mel = torch.log(mel + 1e-6)
        return mel.mean(dim=2)


# ====================================================================
# 端到端联合大模型 (JointTSEASR)
# ====================================================================

class JointTSEASR(nn.Module):
    """端到端 TSE-ASR 联合大模型。

    整合以下子模块：
    1. **Speaker Encoder** — 从唤醒音频中提取目标说话人声纹嵌入
    2. **TSE Extractor** — 基于声纹嵌入从混合音频中提取干净语音
    3. **Rejection Head** — 判断提纯音频中是否真的包含目标说话人
    4. **ASR Backend** — 对提纯后的语音进行识别，输出 CTC logits

    设计要点:
        - 所有模块可在训练时联合优化，也可分别冻结局部参数
        - 输出包含 clean_wavs、reject_logits 和 asr_logits，
          方便外侧的训练脚本使用不同的损失函数组合
        - 长度信息 (enroll_lengths, mixed_lengths) 贯通全流程，
          确保每个模块正确忽略 padding 帧

    Parameters
    ----------
    spk_emb_dim : int, optional
        声纹嵌入维度，默认 256（与 LightSpeakerEncoder 对齐）。
    spk_channels : int, optional
        声纹编码器通道数，默认 256。
    tse_feature_dim : int, optional
        TSE 网络特征维度，默认 256。
    tse_hidden_dim : int, optional
        TSE 网络隐藏层维度，默认 512。
    tse_repeats : int, optional
        TSE TCN 重复次数，默认 2。
    asr_conformer_dim : int, optional
        ASR Conformer 维度，默认 256。
    asr_conformer_layers : int, optional
        ASR Conformer 层数，默认 4（轻量）。
    vocab_size : int, optional
        ASR 输出词表大小，默认 4000。
    n_mels : int, optional
        Mel 滤波器组数量，默认 80。
    """

    def __init__(
        self,
        spk_emb_dim: int = 256,
        spk_channels: int = 256,
        tse_feature_dim: int = 256,
        tse_hidden_dim: int = 512,
        tse_repeats: int = 2,
        asr_conformer_dim: int = 256,
        asr_conformer_layers: int = 4,
        vocab_size: int = 4000,
        n_mels: int = 80,
    ) -> None:
        super().__init__()

        # ---------- 1. 声纹提取器 ----------
        self.speaker_encoder = LightSpeakerEncoder(
            embedding_dim=spk_emb_dim,
            n_mels=n_mels,
            channels=spk_channels,
            sample_rate=16000,
        )

        # ---------- 2. 目标说话人提取器 ----------
        self.tse_extractor = LightTSExtractor(
            in_channels=1,
            out_channels=1,
            feature_dim=tse_feature_dim,
            hidden_dim=tse_hidden_dim,
            emb_dim=spk_emb_dim,
            kernel_size=16,
            stride=8,
            tcn_kernel=3,
            repeats=tse_repeats,
            blocks_per_repeat=5,
        )

        self.fusion_gate = WaveformFusionGate(
            emb_dim=spk_emb_dim,
            hidden_dim=128,
        )

        # ---------- 3. 拒识分类头 ----------
        self.rejection_head = RejectionHead(
            n_mels=n_mels,
            emb_dim=spk_emb_dim,
            hidden_dim=256,
        )

        # ---------- 4. 轻量级 ASR 后端 ----------
        self.asr_backend = LightASRBackend(
            input_dim=n_mels,
            conformer_dim=asr_conformer_dim,
            conformer_layers=asr_conformer_layers,
            conformer_heads=4,
            ffn_dim=1024,
            vocab_size=vocab_size,
            dropout=0.1,
        )

    def forward(
        self,
        enroll_wavs: torch.Tensor,
        enroll_lengths: torch.Tensor,
        mixed_wavs: torch.Tensor,
        mixed_lengths: torch.Tensor,
    ) -> Dict[str, Any]:
        """前向传播：完整的 TSE-ASR 推理流程。

        Parameters
        ----------
        enroll_wavs : torch.Tensor
            唤醒音频波形，shape [B, 1, T_enr]。
        enroll_lengths : torch.Tensor
            唤醒音频真实采样点数，shape [B]。
        mixed_wavs : torch.Tensor
            混合音频波形，shape [B, 1, T_mix]。
        mixed_lengths : torch.Tensor
            混合音频真实采样点数，shape [B]。

        Returns
        -------
        Dict[str, Any]
            包含以下键的字典:
            - ``clean_wavs`` (torch.Tensor): 提纯后波形 [B, 1, T_mix]
            - ``reject_logits`` (torch.Tensor): 拒识二分类 logits [B, 2]
            - ``asr_logits`` (torch.Tensor): ASR CTC logits [B, T_enc, vocab_size]
            - ``asr_lengths`` (torch.Tensor): ASR 编码长度 [B]（用于 CTC loss）
        """
        # ==========================================================
        # Stage 1: Speaker Encoding
        # ==========================================================
        # 从唤醒音频提取目标说话人的声纹嵌入
        speaker_embedding = self.speaker_encoder(
            enroll_wavs, enroll_lengths
        )  # [B, spk_emb_dim]

        # ==========================================================
        # Stage 2: Target Speaker Extraction
        # ==========================================================
        # 利用声纹嵌入从混合音频中提取仅含目标说话人的干净语音
        tse_wavs = self.tse_extractor(
            mixed_wavs, speaker_embedding
        )  # [B, 1, T_mix]
        clean_wavs, fusion_gate = self.fusion_gate(
            mixed_wavs, tse_wavs, speaker_embedding
        )

        # ==========================================================
        # Stage 3: Rejection Classification
        # ==========================================================
        # 判断提纯音频是否真的包含目标说话人 (pos/neg 二分类)
        reject_logits = self.rejection_head(
            clean_wavs,
            speaker_embedding,
            mixed_wavs=mixed_wavs,
            extracted_wavs=tse_wavs,
        )  # [B, 2]

        # ==========================================================
        # Stage 4: ASR Recognition
        # ==========================================================
        # 对提纯后的语音进行识别，输出 CTC 格式的字符级 logits
        asr_logits, asr_lengths = self.asr_backend(
            clean_wavs, mixed_lengths
        )  # [B, T_enc, vocab_size], [B]

        return {
            "clean_wavs": clean_wavs,         # [B, 1, T_mix]
            "tse_wavs": tse_wavs,             # [B, 1, T_mix]
            "fusion_gate": fusion_gate,       # [B, 1, 1]
            "reject_logits": reject_logits,   # [B, 2]
            "asr_logits": asr_logits,         # [B, T_enc, vocab_size]
            "asr_lengths": asr_lengths,       # [B]
        }

    def get_total_params(self) -> Dict[str, int]:
        """统计各子模块的参数量。

        Returns
        -------
        Dict[str, int]
            各模块名称及其参数量的字典。
        """
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "speaker_encoder": count(self.speaker_encoder),
            "tse_extractor": count(self.tse_extractor),
            "fusion_gate": count(self.fusion_gate),
            "rejection_head": count(self.rejection_head),
            "asr_backend": count(self.asr_backend),
        }


# ====================================================================
# 主程序入口（快速自检）
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("JointTSEASR 端到端联合大模型自检")
    print("=" * 60)

    # ---------- 1. 构造模型 ----------
    model = JointTSEASR(
        spk_emb_dim=256,
        spk_channels=256,
        tse_feature_dim=256,
        tse_hidden_dim=512,
        tse_repeats=2,
        asr_conformer_dim=256,
        asr_conformer_layers=4,
        vocab_size=4000,
        n_mels=80,
    )

    # 统计参数量
    param_stats = model.get_total_params()
    total = sum(param_stats.values())
    print(f"\n[参数量统计]")
    for name, num in param_stats.items():
        print(f"  {name:<20s}: {num:>12,}")
    print(f"  {'TOTAL':<20s}: {total:>12,}")

    # ---------- 2. 模拟输入 ----------
    batch_size = 2

    # 唤醒音频: 1.5s 和 2.0s @ 16kHz
    enroll_lengths = torch.tensor([24000, 32000], dtype=torch.long)
    enroll_max = enroll_lengths.max().item()
    enroll_wavs = torch.zeros(batch_size, 1, enroll_max)
    for i, length in enumerate(enroll_lengths):
        enroll_wavs[i, 0, :length] = torch.randn(length) * 0.01

    # 混合音频: 2.0s 和 3.0s @ 16kHz
    mixed_lengths = torch.tensor([32000, 48000], dtype=torch.long)
    mixed_max = mixed_lengths.max().item()
    mixed_wavs = torch.zeros(batch_size, 1, mixed_max)
    for i, length in enumerate(mixed_lengths):
        mixed_wavs[i, 0, :length] = torch.randn(length) * 0.5

    print(f"\n[输入形状]")
    print(f"  enroll_wavs   : {enroll_wavs.shape}")
    print(f"  enroll_lengths: {enroll_lengths.tolist()}")
    print(f"  mixed_wavs    : {mixed_wavs.shape}")
    print(f"  mixed_lengths : {mixed_lengths.tolist()}")

    # ---------- 3. 前向传播 ----------
    model.eval()
    with torch.inference_mode():
        outputs = model(
            enroll_wavs=enroll_wavs,
            enroll_lengths=enroll_lengths,
            mixed_wavs=mixed_wavs,
            mixed_lengths=mixed_lengths,
        )

    print(f"\n[输出结果]")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:<15s}: shape={list(val.shape)}")
        else:
            print(f"  {key:<15s}: type={type(val).__name__}, value={val}")

    # ---------- 4. 验证输出形状 ----------
    assert outputs["clean_wavs"].shape == mixed_wavs.shape, (
        f"clean_wavs 形状不匹配！期望 {mixed_wavs.shape}，"
        f"实际 {outputs['clean_wavs'].shape}"
    )
    assert outputs["reject_logits"].shape == (batch_size, 2), (
        f"reject_logits 形状不匹配！期望 ({batch_size}, 2)，"
        f"实际 {outputs['reject_logits'].shape}"
    )
    assert outputs["asr_logits"].shape[0] == batch_size, (
        f"asr_logits batch 不匹配！期望 {batch_size}，"
        f"实际 {outputs['asr_logits'].shape[0]}"
    )
    assert outputs["asr_logits"].shape[2] == model.asr_backend.vocab_size, (
        f"vocab_size 不匹配！"
    )
    print(f"\n[形状验证] 全部通过 OK")

    # ---------- 5. 调度流程概览 ----------
    print(f"\n[调度流程]")
    print(f"  1. enroll_wavs  →  SpeakerEncoder  →  speaker_embedding  [{batch_size}, 256]")
    print(f"  2. mixed_wavs + spk_emb → TSE Extractor → clean_wavs  {list(outputs['clean_wavs'].shape)}")
    print(f"  3. clean_wavs + spk_emb → RejectionHead → reject_logits  {list(outputs['reject_logits'].shape)}")
    print(f"  4. clean_wavs  →  ASR Backend  →  asr_logits  {list(outputs['asr_logits'].shape)}")

    print(f"\n自检完成！模型总参数量 = {total:,}")
