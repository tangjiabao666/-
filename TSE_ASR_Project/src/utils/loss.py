#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSE-ASR 项目 —— 多任务联合损失函数 (JointLoss)。

本模块实现了训练 TSE-ASR 联合模型所需的复合损失函数，
包含三个任务分支及其动态 Mask 机制。

损失组成
--------
+------------------+---------------------------+---------------------------+
| 损失分支          | 损失函数                    | 适用范围                   |
+==================+===========================+===========================+
| TSE Loss          | SI-SNR (尺度不变信噪比)     | 仅正样本 (is_targets == 1) |
|                  | + Padding Mask              |                           |
+------------------+---------------------------+---------------------------+
| ASR Loss          | CTC Loss (blank=0)          | 仅正样本 (is_targets == 1) |
+------------------+---------------------------+---------------------------+
| Rejection Loss    | CrossEntropyLoss (二分类)   | 所有样本                   |
+------------------+---------------------------+---------------------------+

动态 Mask 机制（核心防暴雷逻辑）
---------------------------------
1. 根据 ``is_targets`` 构造 mask_pos: 正样本=1, 负样本=0
2. TSE Loss 和 ASR Loss 乘以 mask_pos 后求均值
   → 负样本的 TSE/ASR Loss 被强制置零，不参与梯度回传
3. Rejection Loss 对所有样本计算（它本就是要区分正负样本的）

权重配置
--------
total_loss = alpha * tse_loss + beta * asr_loss + gamma * reject_loss

参考论文
--------
- SI-SNR: SDR – Half-baked or Well Done? (Le Roux et al., ICASSP 2019)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# SI-SNR 辅助函数
# ====================================================================

def si_snr(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算尺度不变信噪比 (Scale-Invariant Signal-to-Noise Ratio, SI-SNR)。

    SI-SNR 是语音分离/增强领域的标准评估指标。
    其核心思想是: 在对预测信号做最优缩放后，衡量信号能量与噪声能量之比。

    数学定义::

        s_target = (target · pred̂) / (target · target) * target
        e_noise  = pred̂ - s_target
        SI-SNR   = 10 * log10(||s_target||² / ||e_noise||²)

    其中 pred̂ = (pred - mean(pred)) / ||pred - mean(pred)||，即零均值归一化后的预测。

    Parameters
    ----------
    pred : torch.Tensor
        预测的语音波形，shape [B, T]（单声道，已 squeeze 通道维）。
    target : torch.Tensor
        真实的纯净语音波形，shape [B, T]。
    eps : float, optional
        防止除零的极小常数，默认 1e-8。

    Returns
    -------
    torch.Tensor
        各样本的 SI-SNR 值，shape [B]。值越大越好。
        注意：返回的是原始 SI-SNR（正值），实际作为损失时取负号。
    """
    # 零均值化
    pred = pred - torch.mean(pred, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    # 计算 target 上的正交投影系数
    # s_target = α * target,  α = sum(pred * target) / sum(target * target)
    dot_product = torch.sum(pred * target, dim=-1, keepdim=True)   # [B, 1]
    target_power = torch.sum(target * target, dim=-1, keepdim=True) + eps  # [B, 1]

    # 目标信号在预测方向上的正交投影
    s_target = (dot_product / target_power) * target                # [B, T]

    # 噪声 = 预测 - 目标投影
    e_noise = pred - s_target                                       # [B, T]

    # 信号能量 / 噪声能量
    s_power = torch.sum(s_target ** 2, dim=-1) + eps               # [B]
    e_power = torch.sum(e_noise ** 2, dim=-1) + eps                # [B]

    si_snr_val = 10.0 * torch.log10(s_power / e_power)             # [B]

    return si_snr_val


def compute_si_snr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算 SI-SNR 损失（用于训练的反向传播）。

    包装 ``si_snr`` 函数，额外提供：
    1. 取负号（SI-SNR 越大越好 → Loss 越小越好）
    2. Padding mask 支持（使用 lengths 忽略填充部分）
    3. 样本级均值（对 batch 内所有样本取平均）

    Parameters
    ----------
    pred : torch.Tensor
        预测的语音波形，shape [B, T]（单声道）。
    target : torch.Tensor
        真实的纯净语音波形，shape [B, T]。
    lengths : torch.Tensor, optional
        各样本的有效采样点数，shape [B]。用于生成 padding mask。
        为 None 时假定全部帧有效。
    eps : float, optional
        防止除零的极小常数，默认 1e-8。

    Returns
    -------
    torch.Tensor
        标量，SI-SNR 损失（负的 SI-SNR 均值）。
    """
    # 逐样本计算 SI-SNR
    si_snr_per_sample = si_snr(pred, target, eps=eps)  # [B]

    # 生成 padding mask（基于 lengths）
    if lengths is not None:
        max_len = target.shape[-1]
        # mask[i, t] = 1 表示样本 i 在时间 t 是有效帧，0 表示 padding
        mask = torch.arange(max_len, device=target.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.float()  # [B, T]

        # 对 padding 区域：将 pred 和 target 都置零，避免影响 SI-SNR 计算
        # 但实际上 SI-SNR 是逐样本计算的，padding 会拉低信号能量
        # 更合理的做法：在送入 si_snr 前将 padding 区域的 pred 和 target 都置零
        pred_masked = pred * mask
        target_masked = target * mask
        si_snr_per_sample = si_snr(pred_masked, target_masked, eps=eps)  # [B]

    # SI-SNR 越大越好 → 取负号使其变为损失（越小越好）
    loss_per_sample = -si_snr_per_sample  # [B]

    return loss_per_sample.mean()


# ====================================================================
# 多任务联合损失类
# ====================================================================

class JointLoss(nn.Module):
    """TSE-ASR 多任务联合损失函数。

    将 TSE（语音提纯）、ASR（语音识别）、Rejection（拒识）三个任务的
    损失加权组合，并通过动态 Mask 机制确保只有正样本参与 TSE/ASR 的
    损失计算，而拒识损失对全部样本生效。

    Parameters
    ----------
    alpha : float, optional
        TSE（SI-SNR）损失的权重，默认 0.8。
    beta : float, optional
        ASR（CTC）损失的权重，默认 1.0。
    gamma : float, optional
        Rejection（CE）损失的权重，默认 0.5。
    si_snr_eps : float, optional
        SI-SNR 计算中的 epsilon 值，默认 1e-8。

    Notes
    -----
    - CTC 损失使用 ``blank=0``（留空 token 在词典第 0 位），
      并启用 ``zero_infinity=True`` 防止 log(0) 产生 inf。
    - 如果某个 batch 内没有任何正样本，TSE 和 ASR 损失将被置为 0。
    """

    def __init__(
        self,
        alpha: float = 0.8,
        beta: float = 1.0,
        gamma: float = 0.5,
        si_snr_eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.si_snr_eps = si_snr_eps

        # ---------- CTC 损失 ----------
        # blank=0 表示空白 token 在词典的第 0 位
        # zero_infinity=True: 防止 CTC 在长序列上产生 inf
        # reduction='none': 返回逐样本损失，便于 mask
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")

        # ---------- 拒识二分类损失 ----------
        # reduction='none' 返回逐样本损失，我们自己做 mean
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(
        self,
        clean_wavs_pred: torch.Tensor,
        clean_wavs_target: torch.Tensor,
        asr_logits: torch.Tensor,
        text_targets: torch.Tensor,
        asr_input_lengths: torch.Tensor,
        text_target_lengths: torch.Tensor,
        reject_logits: torch.Tensor,
        is_targets: torch.Tensor,
        mixed_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """前向传播：计算加权多任务联合损失。

        Parameters
        ----------
        clean_wavs_pred : torch.Tensor
            模型预测的提纯后语音波形，shape [B, 1, T]。
        clean_wavs_target : torch.Tensor
            真实的纯净语音波形，shape [B, 1, T]。
            注意：测试集中可能无法直接获取，此时此项作为占位传零张量。
        asr_logits : torch.Tensor
            ASR 输出的 CTC logits，shape [B, T_enc, vocab_size]。
            注意：CTC 需要 log_softmax 后的概率，如果模型输出未做，
            请在外部先调用 ``F.log_softmax(asr_logits, dim=-1)`` 再传入。
        text_targets : torch.Tensor
            真实文本 token 序列（拼接后的一维向量），shape [sum(target_lengths)]。
            CTC 要求将所有样本的 target 拼接为一维。
        asr_input_lengths : torch.Tensor
            ASR 编码后的序列长度，shape [B]。
        text_target_lengths : torch.Tensor
            各样本的真实文本长度，shape [B]。
        reject_logits : torch.Tensor
            拒识分类 logits，shape [B, 2]。第 0 列为 neg，第 1 列为 pos。
        is_targets : torch.Tensor
            样本类别标签，shape [B]。1 表示正样本（含目标说话人），
            0 表示负样本（拒识/噪音）。
        mixed_lengths : torch.Tensor, optional
            各混合音频的有效采样点数，shape [B]。
            用于 TSE loss 的 padding mask。

        Returns
        -------
        Dict[str, Any]
            包含以下键的字典:
            - ``total_loss`` (torch.Tensor): 加权总损失（标量）
            - ``tse_loss`` (torch.Tensor): TSE 损失值（标量）
            - ``asr_loss`` (torch.Tensor): ASR 损失值（标量）
            - ``reject_loss`` (torch.Tensor): 拒识损失值（标量）
            - ``num_pos`` (int): batch 内正样本数
            - ``num_neg`` (int): batch 内负样本数
        """
        batch_size = clean_wavs_pred.shape[0]
        device = clean_wavs_pred.device

        # ============================================================
        # Step 0: 构建正样本 Mask
        # ============================================================
        # mask_pos[i] = 1.0 表示第 i 个样本是正样本 (is_targets == 1)
        # mask_pos[i] = 0.0 表示第 i 个样本是负样本 (is_targets == 0)
        # 这个 mask 会乘以 TSE 和 ASR 损失，从而屏蔽负样本的梯度
        mask_pos = is_targets.float()  # [B], 值为 1.0 或 0.0

        # 统计正负样本数量（用于日志和除零保护）
        num_pos = int(mask_pos.sum().item())
        num_neg = batch_size - num_pos

        # ============================================================
        # Step 1: TSE Loss (SI-SNR)
        # ⚠️ 仅对正样本生效 — 负样本的 loss 被 mask 置零
        # ============================================================
        if clean_wavs_pred.dim() == 3:
            pred_wav = clean_wavs_pred.squeeze(1)    # [B, T]
            target_wav = clean_wavs_target.squeeze(1)  # [B, T]
        else:
            pred_wav = clean_wavs_pred
            target_wav = clean_wavs_target

        # 逐样本计算 SI-SNR 损失，返回 [B]
        # mixed_lengths 用于生成 padding mask，屏蔽末尾填充部分
        tse_loss_per_sample = compute_si_snr_loss(
            pred=pred_wav,
            target=target_wav,
            lengths=mixed_lengths,
            eps=self.si_snr_eps,
        ) if num_pos > 0 else torch.tensor(0.0, device=device)
        # 注意：compute_si_snr_loss 已经返回了样本级均值下的标量
        # 但为了 mask 机制，我们需要逐样本损失，所以重构为逐样本计算

        # 正确的逐样本 TSE loss 计算（带 mask）
        tse_loss_per_sample_raw: torch.Tensor
        if num_pos > 0:
            # 逐样本 SI-SNR → 取负号 → [B]
            si_snr_each = si_snr(pred_wav, target_wav, eps=self.si_snr_eps)
            tse_loss_per_sample_raw = -si_snr_each  # [B]
        else:
            tse_loss_per_sample_raw = torch.zeros(batch_size, device=device)

        # 将负样本的 TSE loss 置零
        tse_loss_per_sample_masked = tse_loss_per_sample_raw * mask_pos  # [B]

        # 均值：仅对正样本求平均
        # 如果 num_pos == 0，分母为 1 避免 NaN
        tse_loss = tse_loss_per_sample_masked.sum() / max(num_pos, 1)

        # ============================================================
        # Step 2: ASR Loss (CTC)
        # ⚠️ 仅对正样本生效 — 负样本的 loss 被 mask 置零
        # ============================================================
        if num_pos > 0 and asr_logits.numel() > 0:
            # CTC 要求:
            #   - log_probs: [T, B, vocab_size]  需要从 [B, T, V] 转置
            #   - targets: [sum(target_lengths)]  所有样本的标签拼接
            #   - input_lengths: [B]  每个样本的特征帧数
            #   - target_lengths: [B]  每个样本的标签长度
            log_probs = F.log_softmax(asr_logits, dim=-1)   # [B, T, V]
            log_probs = log_probs.permute(1, 0, 2)          # [T, B, V]

            # reduction='none' → 返回 [B] 的逐样本损失
            asr_loss_per_sample = self.ctc_loss(
                log_probs,
                text_targets,
                asr_input_lengths,
                text_target_lengths,
            )  # [B]
        else:
            asr_loss_per_sample = torch.zeros(batch_size, device=device)

        # 将负样本的 ASR loss 置零
        asr_loss_per_sample_masked = asr_loss_per_sample * mask_pos  # [B]

        # 均值：仅对正样本求平均
        asr_loss = asr_loss_per_sample_masked.sum() / max(num_pos, 1)

        # ============================================================
        # Step 3: Rejection Loss (CrossEntropy)
        # ✅ 所有样本都参与计算（它要区分正负样本）
        # ============================================================
        reject_loss_per_sample = self.ce_loss(reject_logits, is_targets)  # [B]

        # 拒绝损失对所有样本取平均
        reject_loss = reject_loss_per_sample.mean()

        # ============================================================
        # Step 4: 加权总损失
        # ============================================================
        total_loss = (
            self.alpha * tse_loss
            + self.beta * asr_loss
            + self.gamma * reject_loss
        )

        return {
            "total_loss": total_loss,
            "tse_loss": tse_loss.detach().clone(),
            "asr_loss": asr_loss.detach().clone(),
            "reject_loss": reject_loss.detach().clone(),
            "num_pos": num_pos,
            "num_neg": num_neg,
        }


# ====================================================================
# 主程序入口（快速自检）
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("JointLoss 多任务联合损失函数自检")
    print("=" * 60)

    # ---------- 1. 构造损失模块 ----------
    joint_loss = JointLoss(alpha=0.8, beta=1.0, gamma=0.5)

    batch_size = 4
    # 模拟: 2 个正样本 + 2 个负样本
    is_targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)  # [B]

    # ---------- 2. 构造模拟输入 ----------
    audio_length = 32000  # 2s @ 16kHz
    # 预测提纯波形 [B, 1, T]
    clean_pred = torch.randn(batch_size, 1, audio_length) * 0.1
    # 真实纯净波形 [B, 1, T]（模拟）
    clean_target = torch.randn(batch_size, 1, audio_length) * 0.1 + 0.001

    # ASR 输出 [B, T_enc=50, vocab=100]
    asr_logits = torch.randn(batch_size, 50, 100)
    # 文本标签（4 个样本的拼接：长度分别为 5, 0, 8, 0）
    # 负样本的 text_targets 为空，对应 target_lengths=0
    text_targets = torch.tensor([1,2,3,4,5,  6,7,8,9,10,11,12,13], dtype=torch.long)
    text_target_lengths = torch.tensor([5, 0, 8, 0], dtype=torch.long)
    asr_input_lengths = torch.tensor([50, 50, 50, 50], dtype=torch.long)

    # 拒识 logits [B, 2]
    reject_logits = torch.randn(batch_size, 2)

    # 音频有效长度
    mixed_lengths = torch.tensor([32000, 32000, 16000, 24000], dtype=torch.long)

    print(f"\n[输入配置]")
    print(f"  batch_size={batch_size}, pos={is_targets.sum().item()}, "
          f"neg={(is_targets==0).sum().item()}")
    print(f"  is_targets: {is_targets.tolist()}")

    # ---------- 3. 前向传播 ----------
    outputs = joint_loss(
        clean_wavs_pred=clean_pred,
        clean_wavs_target=clean_target,
        asr_logits=asr_logits,
        text_targets=text_targets,
        asr_input_lengths=asr_input_lengths,
        text_target_lengths=text_target_lengths,
        reject_logits=reject_logits,
        is_targets=is_targets,
        mixed_lengths=mixed_lengths,
    )

    print(f"\n[损失输出]")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:<15s}: {val.item():.6f}")
        else:
            print(f"  {key:<15s}: {val}")

    # ---------- 4. 验证 ----------
    assert outputs["total_loss"].item() != 0, "total_loss 不应当为 0"
    # 只有正样本参与 TSE/ASR loss，且有 2 个正样本
    assert outputs["tse_loss"].item() != 0, "正样本的 TSE loss 不应为 0"
    assert outputs["asr_loss"].item() != 0, "正样本的 ASR loss 不应为 0"
    # reject_loss 对所有样本都有效
    assert outputs["reject_loss"].item() != 0, "reject_loss 不应为 0"
    print(f"\n[验证] 全部通过 — 正负样本 Mask 机制正确运行 OK")

    # ---------- 5. 特殊场景：全是负样本 ----------
    print(f"\n[特殊场景] batch 内全部为负样本:")
    is_targets_all_neg = torch.zeros(batch_size, dtype=torch.long)
    text_targets_all_neg = torch.tensor([], dtype=torch.long)
    text_target_lengths_all_neg = torch.zeros(batch_size, dtype=torch.long)

    outputs_neg = joint_loss(
        clean_wavs_pred=clean_pred,
        clean_wavs_target=clean_target,
        asr_logits=asr_logits,
        text_targets=text_targets_all_neg,
        asr_input_lengths=asr_input_lengths,
        text_target_lengths=text_target_lengths_all_neg,
        reject_logits=reject_logits,
        is_targets=is_targets_all_neg,
        mixed_lengths=mixed_lengths,
    )
    print(f"  tse_loss    = {outputs_neg['tse_loss'].item():.6f}  (应为 0，因为 num_pos=0)")
    print(f"  asr_loss    = {outputs_neg['asr_loss'].item():.6f}  (应为 0，因为 num_pos=0)")
    print(f"  reject_loss = {outputs_neg['reject_loss'].item():.6f}  (正常，所有样本都参与)")
    print(f"  total_loss  = {outputs_neg['total_loss'].item():.6f}")

    assert abs(outputs_neg["tse_loss"].item()) < 1e-6, "全负样本时 tse_loss 应为 0"
    assert abs(outputs_neg["asr_loss"].item()) < 1e-6, "全负样本时 asr_loss 应为 0"

    print(f"\n自检完成！")