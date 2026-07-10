#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSE-ASR 项目 —— 端到端联合训练主入口。

本脚本将数据加载、联合模型、多任务损失函数整合为一个完整的训练循环，
支持 AMP 混合精度训练、梯度裁剪、学习率余弦退火与断点续训。

使用示例
--------
.. code-block:: bash

    # 基础训练
    python scripts/train.py \
        --jsonl_path data/raw/test_set_a/labels.jsonl \
        --audio_dir data/raw/test_set_a \
        --batch_size 4 \
        --epochs 50 \
        --lr 1e-4

    # 从断点恢复
    python scripts/train.py \
        --jsonl_path data/raw/test_set_a/labels.jsonl \
        --audio_dir data/raw/test_set_a \
        --resume checkpoints/epoch_10.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader

# 将项目根目录加入路径，确保 src 包可以被导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import ComplexSpeechDataset, create_dataloader_from_config
from src.models.joint_model import JointTSEASR
from src.utils.loss import JointLoss

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ====================================================================
# 简单字符级 Tokenizer（占位符方案）
# ====================================================================

class SimpleCharTokenizer:
    """简单的字符级 Tokenizer，用于将中文字符串转换为 token ID 序列。

    设计说明
    --------
    在真实的工业级项目中会使用 SentencePiece / BPE / WordPiece 等
    子词切分方案，并加载预训练好的词典。
    此处为了快速跑通整个训练流水线，采用最直接的"按字符切分 + 动态建表"
    方案：
        - 遍历训练集所有文本标签，收集出现的所有字符
        - 建立 char → id 的映射字典
        - 预留 id=0 作为 CTC 的 <blank> token
        - 未登录字符映射到 id=0

    Attributes
    ----------
    blank_id : int
        CTC 空白 token 的 ID，固定为 0。
    char_to_id : Dict[str, int]
        字符到 ID 的映射字典。
    id_to_char : Dict[int, str]
        ID 到字符的映射字典（仅用于调试/解码）。
    vocab_size : int
        词典大小（含 blank）。
    """

    def __init__(self) -> None:
        # id=0 预留给 CTC blank token
        self.blank_id: int = 0
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size: int = 1  # 至少包含 blank

    def build_from_texts(self, texts: List[str]) -> None:
        """从文本标签列表中构建字符级词典。

        遍历所有文本，收集出现的唯一字符，按发现顺序分配 ID。

        Parameters
        ----------
        texts : List[str]
            训练集中的所有文本标签。
        """
        # 重置为初始状态
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 1  # id=0 是 blank

        unique_chars: List[str] = []
        seen: set = set()
        for text in texts:
            for ch in text:
                if ch not in seen:
                    seen.add(ch)
                    unique_chars.append(ch)

        # 从 id=1 开始分配字符 ID
        for idx, ch in enumerate(unique_chars, start=1):
            self.char_to_id[ch] = idx
            self.id_to_char[idx] = ch

        self.vocab_size = len(self.char_to_id) + 1  # +1 是 blank

        logger.info(
            "词典构建完成: vocab_size=%d (含 blank), 字符数=%d",
            self.vocab_size,
            len(unique_chars),
        )

    def encode(self, text: str) -> List[int]:
        """将文本字符串编码为 token ID 列表。

        Parameters
        ----------
        text : str
            待编码的文本。

        Returns
        -------
        List[int]
            token ID 列表。空字符串返回空列表。
        """
        if not text or len(text.strip()) == 0:
            return []
        return [self.char_to_id.get(ch, self.blank_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """将 token ID 列表解码回文本（用于调试）。

        Parameters
        ----------
        ids : List[int]
            token ID 列表。

        Returns
        -------
        str
            解码后的文本。
        """
        return "".join(
            self.id_to_char.get(i, "<unk>") for i in ids if i != self.blank_id
        )


def build_tokenizer_from_dataset(
    jsonl_path: str,
    root_dir: str,
) -> SimpleCharTokenizer:
    """从数据集构建字符级 Tokenizer。

    Parameters
    ----------
    jsonl_path : str
        JSONL 标注文件路径。
    root_dir : str
        音频根目录。

    Returns
    -------
    SimpleCharTokenizer
        构建好的 tokenizer 实例。
    """
    # 创建临时数据集，收集所有文本标签
    ds = ComplexSpeechDataset(
        jsonl_path=jsonl_path,
        root_dir=root_dir,
        sample_rate=16000,
    )

    # 收集所有非空文本标签
    texts: List[str] = []
    for i in range(len(ds)):
        sample = ds[i]
        label = sample.get("text_label", "")
        if label:
            texts.append(label)

    logger.info("共收集到 %d 条非空文本标签用于构建词典", len(texts))

    tokenizer = SimpleCharTokenizer()
    tokenizer.build_from_texts(texts)
    return tokenizer


# ====================================================================
# 批次文本处理辅助函数
# ====================================================================

def collate_text_batch(
    text_labels: List[str],
    tokenizer: SimpleCharTokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将一批文本标签转换为 CTC Loss 所需的格式。

    CTC 要求:
        - targets: 所有样本标签拼接为 1D Tensor, shape [sum(target_lengths)]
        - target_lengths: Tensor, shape [B]

    处理流程:
        1. 对每段文本调用 tokenizer.encode() 得到 token IDs
        2. 拼接所有 token IDs 为 1D targets
        3. 记录各样本的 target_lengths

    Parameters
    ----------
    text_labels : List[str]
        一批文本标签，长度 = batch_size。
    tokenizer : SimpleCharTokenizer
        字符级 tokenizer 实例。
    device : torch.device
        目标设备。

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - targets: [sum(target_lengths)], dtype=torch.long
        - target_lengths: [B], dtype=torch.long
    """
    encoded_batch: List[List[int]] = []
    target_lengths_list: List[int] = []

    for text in text_labels:
        token_ids = tokenizer.encode(text)
        encoded_batch.append(token_ids)
        target_lengths_list.append(len(token_ids))

    # 拼接所有 token IDs
    if len(encoded_batch) > 0 and any(len(ids) > 0 for ids in encoded_batch):
        targets = torch.tensor(
            [tid for ids in encoded_batch for tid in ids],
            dtype=torch.long,
            device=device,
        )
    else:
        targets = torch.tensor([], dtype=torch.long, device=device)

    target_lengths = torch.tensor(target_lengths_list, dtype=torch.long, device=device)

    return targets, target_lengths


# ====================================================================
# 训练主循环
# ====================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: JointLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    tokenizer: SimpleCharTokenizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
    max_grad_norm: float = 5.0,
) -> Dict[str, float]:
    """执行一个 epoch 的训练。

    Parameters
    ----------
    model : nn.Module
        JointTSEASR 联合模型。
    dataloader : DataLoader
        训练数据加载器。
    loss_fn : JointLoss
        联合损失函数。
    optimizer : torch.optim.Optimizer
        优化器。
    scheduler : _LRScheduler or None
        学习率调度器（在每个 batch 后 step）。
    scaler : GradScaler
        AMP 梯度缩放器。
    tokenizer : SimpleCharTokenizer
        字符级 tokenizer。
    device : torch.device
        训练设备。
    epoch : int
        当前 epoch 编号（从 1 开始）。
    log_interval : int, optional
        每隔多少 batch 打印一次日志，默认 50。
    max_grad_norm : float, optional
        梯度裁剪的最大范数，默认 5.0。

    Returns
    -------
    Dict[str, float]
        当前 epoch 的平均损失统计。
    """
    model.train()
    total_loss_sum = 0.0
    tse_loss_sum = 0.0
    asr_loss_sum = 0.0
    reject_loss_sum = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # ---------- Step 1: 数据转移到 GPU ----------
        enroll_wavs = batch["enroll_wavs"].to(device, non_blocking=True)      # [B, 1, T_enr]
        enroll_lengths = batch["enroll_lengths"].to(device, non_blocking=True)  # [B]
        mixed_wavs = batch["mixed_wavs"].to(device, non_blocking=True)          # [B, 1, T_mix]
        mixed_lengths = batch["mixed_lengths"].to(device, non_blocking=True)    # [B]
        text_labels: List[str] = batch["text_labels"]                           # List[str]
        is_targets = batch["is_targets"].to(device, non_blocking=True)          # [B]

        # ---------- Step 2: 文本 → Token 转换 ----------
        text_targets, text_target_lengths = collate_text_batch(
            text_labels, tokenizer, device
        )

        # ---------- Step 3: 前向传播 (AMP) ----------
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(
                enroll_wavs=enroll_wavs,
                enroll_lengths=enroll_lengths,
                mixed_wavs=mixed_wavs,
                mixed_lengths=mixed_lengths,
            )
            # outputs: clean_wavs, reject_logits, asr_logits, asr_lengths

            # ---------- Step 4: 计算损失 ----------
            loss_dict = loss_fn(
                clean_wavs_pred=outputs["clean_wavs"],
                clean_wavs_target=mixed_wavs,  # 测试集无纯净标注，用 mixed 作为占位
                asr_logits=outputs["asr_logits"],
                text_targets=text_targets,
                asr_input_lengths=outputs["asr_lengths"],
                text_target_lengths=text_target_lengths,
                reject_logits=outputs["reject_logits"],
                is_targets=is_targets,
                mixed_lengths=mixed_lengths,
            )

        loss = loss_dict["total_loss"]

        # ---------- Step 5: 反向传播 + 梯度裁剪 + 优化器更新 ----------
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # 梯度裁剪：防止梯度爆炸
        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        # 学习率调度（如果有）
        if scheduler is not None:
            scheduler.step()

        # ---------- Step 6: 累积统计 ----------
        total_loss_sum += loss_dict["total_loss"].item()
        tse_loss_sum += loss_dict["tse_loss"].item()
        asr_loss_sum += loss_dict["asr_loss"].item()
        reject_loss_sum += loss_dict["reject_loss"].item()
        num_batches += 1

        # ---------- Step 7: 日志输出 ----------
        if (batch_idx + 1) % log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d | Batch %d/%d | "
                "Total=%.4f | TSE=%.4f | ASR=%.4f | Rej=%.4f | "
                "pos=%d neg=%d | LR=%.2e",
                epoch,
                batch_idx + 1,
                len(dataloader),
                loss_dict["total_loss"].item(),
                loss_dict["tse_loss"].item(),
                loss_dict["asr_loss"].item(),
                loss_dict["reject_loss"].item(),
                loss_dict["num_pos"],
                loss_dict["num_neg"],
                current_lr,
            )

    # 计算 epoch 平均
    avg_total = total_loss_sum / max(num_batches, 1)
    avg_tse = tse_loss_sum / max(num_batches, 1)
    avg_asr = asr_loss_sum / max(num_batches, 1)
    avg_rej = reject_loss_sum / max(num_batches, 1)

    return {
        "total": avg_total,
        "tse": avg_tse,
        "asr": avg_asr,
        "reject": avg_rej,
    }


# ====================================================================
# 主函数
# ====================================================================

def main(args: argparse.Namespace) -> None:
    """训练主函数。

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数。
    """
    # ---------- 1. 设备检测 ----------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("使用 GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning("未检测到 GPU，使用 CPU 训练（速度会较慢）")

    # ---------- 2. 构建 Tokenizer ----------
    logger.info("正在从训练集构建字符级词典...")
    tokenizer = build_tokenizer_from_dataset(
        jsonl_path=args.jsonl_path,
        root_dir=args.audio_dir,
    )
    logger.info("Tokenizer 词典大小: %d", tokenizer.vocab_size)

    # ---------- 3. 构建 DataLoader ----------
    logger.info("正在创建 DataLoader...")
    dataloader = create_dataloader_from_config(
        jsonl_path=args.jsonl_path,
        root_dir=args.audio_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        sample_rate=16000,
    )
    logger.info("DataLoader 创建完成，共 %d 个 batch", len(dataloader))

    # ---------- 4. 构建模型 ----------
    logger.info("正在构建 JointTSEASR 联合模型...")
    model = JointTSEASR(
        spk_emb_dim=256,
        spk_channels=256,
        tse_feature_dim=256,
        tse_hidden_dim=512,
        tse_repeats=2,
        asr_conformer_dim=256,
        asr_conformer_layers=4,
        vocab_size=tokenizer.vocab_size,  # 使用实际词典大小
        n_mels=80,
    ).to(device)

    # 统计参数量
    param_stats = model.get_total_params()
    total_params = sum(param_stats.values())
    logger.info("模型参数量: %s", {k: f"{v:,}" for k, v in param_stats.items()})
    logger.info("总参数量: %s", f"{total_params:,}")

    # ---------- 5. 损失函数 ----------
    loss_fn = JointLoss(
        alpha=args.tse_weight,
        beta=args.asr_weight,
        gamma=args.reject_weight,
    )

    # ---------- 6. 优化器 ----------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
    )
    logger.info("优化器: AdamW, lr=%.2e, weight_decay=1e-5", args.lr)

    # ---------- 7. 学习率调度器 ----------
    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.01,
    )
    logger.info("学习率调度器: CosineAnnealingLR, T_max=%d, eta_min=%.2e", total_steps, args.lr * 0.01)

    # ---------- 8. AMP 混合精度 ----------
    # CPU 上不使用 AMP
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger.info("AMP 混合精度: %s", "启用" if use_amp else "禁用（CPU 模式）")

    # ---------- 9. 断点恢复 ----------
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("从断点恢复: %s", args.resume)
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            logger.info("恢复成功，从 epoch %d 继续训练", start_epoch)
        else:
            logger.error("断点文件不存在: %s", args.resume)
            return

    # ---------- 10. 创建 Checkpoint 目录 ----------
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 11. 训练循环 ----------
    logger.info("=" * 60)
    logger.info("开始训练！共 %d 个 epoch", args.epochs)
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info("-------- Epoch %d / %d --------", epoch, args.epochs)

        avg_losses = train_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer=tokenizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            max_grad_norm=args.max_grad_norm,
        )

        logger.info(
            "Epoch %d 平均损失: Total=%.4f | TSE=%.4f | ASR=%.4f | Rej=%.4f",
            epoch,
            avg_losses["total"],
            avg_losses["tse"],
            avg_losses["asr"],
            avg_losses["reject"],
        )

        # ---------- 保存 Checkpoint ----------
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "tokenizer_char_to_id": tokenizer.char_to_id,
                "avg_losses": avg_losses,
            },
            checkpoint_path,
        )
        logger.info("Checkpoint 已保存: %s", checkpoint_path)

        # 额外保存最新 checkpoint 的副本
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "tokenizer_char_to_id": tokenizer.char_to_id,
                "avg_losses": avg_losses,
            },
            latest_path,
        )

    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)


# ====================================================================
# 命令行参数解析
# ====================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="TSE-ASR 端到端联合模型训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------- 数据参数 ----------
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="data/raw/test_set_a/labels.jsonl",
        help="JSONL 标注文件路径",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/raw/test_set_a",
        help="音频文件根目录",
    )

    # ---------- 训练超参数 ----------
    parser.add_argument(
        "--batch_size", type=int, default=4, help="批次大小"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="训练轮数"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="初始学习率"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="DataLoader 子进程数"
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="日志打印间隔 (batch)"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=5.0, help="梯度裁剪最大范数"
    )

    # ---------- 损失权重 ----------
    parser.add_argument(
        "--tse_weight", type=float, default=0.8, help="TSE (SI-SNR) 损失权重"
    )
    parser.add_argument(
        "--asr_weight", type=float, default=1.0, help="ASR (CTC) 损失权重"
    )
    parser.add_argument(
        "--reject_weight", type=float, default=0.5, help="Rejection (CE) 损失权重"
    )

    # ---------- 断点续训 ----------
    parser.add_argument(
        "--resume", type=str, default=None, help="断点文件路径 (如 checkpoints/epoch_10.pt)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="模型保存目录"
    )

    return parser.parse_args()


# ====================================================================
# 程序入口
# ====================================================================

if __name__ == "__main__":
    args = parse_args()
    logger.info("训练参数: %s", vars(args))
    main(args)