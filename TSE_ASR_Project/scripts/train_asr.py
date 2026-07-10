#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train only the ASR backend on positive command utterances."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.train import (  # noqa: E402
    SimpleCharTokenizer,
    build_tokenizer_from_dataset,
    collate_text_batch,
)
from src.data.dataset import create_dataloader_from_config  # noqa: E402
from src.models.joint_model import JointTSEASR  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def restore_tokenizer(checkpoint: Dict[str, object]) -> Optional[SimpleCharTokenizer]:
    char_to_id = checkpoint.get("tokenizer_char_to_id")
    if not isinstance(char_to_id, dict):
        return None
    tokenizer = SimpleCharTokenizer()
    tokenizer.char_to_id = {str(ch): int(idx) for ch, idx in char_to_id.items()}
    tokenizer.id_to_char = {idx: ch for ch, idx in tokenizer.char_to_id.items()}
    tokenizer.vocab_size = int(checkpoint.get("tokenizer_vocab_size", len(tokenizer.char_to_id) + 1))
    return tokenizer


def build_model(vocab_size: int, device: torch.device) -> JointTSEASR:
    model = JointTSEASR(
        spk_emb_dim=256,
        spk_channels=256,
        tse_feature_dim=256,
        tse_hidden_dim=512,
        tse_repeats=2,
        asr_conformer_dim=256,
        asr_conformer_layers=4,
        vocab_size=vocab_size,
        n_mels=80,
    )
    return model.to(device)


def set_asr_only_trainable(model: JointTSEASR) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.asr_backend.parameters():
        param.requires_grad = True


def save_checkpoint(
    path: Path,
    epoch: int,
    model: JointTSEASR,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    tokenizer: SimpleCharTokenizer,
    avg_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_char_to_id": tokenizer.char_to_id,
            "avg_losses": {"asr": avg_loss, "total": avg_loss, "tse": 0.0, "reject": 0.0},
            "training_mode": "asr_only_mixed",
        },
        path,
    )


def train_one_epoch(
    model: JointTSEASR,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: SimpleCharTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    log_interval: int,
    max_grad_norm: float,
) -> float:
    model.train()
    model.asr_backend.train()
    ctc_loss = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    loss_sum = 0.0
    steps = 0

    for batch_idx, batch in enumerate(dataloader):
        is_targets = batch["is_targets"].bool()
        if not bool(is_targets.any()):
            continue

        mixed_wavs = batch["mixed_wavs"][is_targets].to(device, non_blocking=True)
        mixed_lengths = batch["mixed_lengths"][is_targets].to(device, non_blocking=True)
        max_len = int(mixed_lengths.max().item())
        mixed_wavs = mixed_wavs[..., :max_len]
        text_labels: List[str] = [
            label for label, keep in zip(batch["text_labels"], is_targets.tolist()) if keep
        ]
        text_targets, text_target_lengths = collate_text_batch(text_labels, tokenizer, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            asr_logits, asr_lengths = model.asr_backend(mixed_wavs, mixed_lengths)
            log_probs = F.log_softmax(asr_logits, dim=-1).transpose(0, 1)
            loss = ctc_loss(log_probs, text_targets, asr_lengths, text_target_lengths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.asr_backend.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        loss_sum += float(loss.detach().item())
        steps += 1

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                "Epoch %d | Batch %d/%d | ASR=%.4f | pos=%d | LR=%.2e",
                epoch,
                batch_idx + 1,
                len(dataloader),
                float(loss.detach().item()),
                len(text_labels),
                optimizer.param_groups[0]["lr"],
            )

    return loss_sum / max(steps, 1)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    checkpoint = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        tokenizer = restore_tokenizer(checkpoint)
        if tokenizer is None:
            tokenizer = build_tokenizer_from_dataset(args.jsonl_path, args.audio_dir)
    else:
        tokenizer = build_tokenizer_from_dataset(args.jsonl_path, args.audio_dir)

    dataloader = create_dataloader_from_config(
        jsonl_path=args.jsonl_path,
        root_dir=args.audio_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        sample_rate=16000,
    )

    model = build_model(tokenizer.vocab_size, device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("已加载模型权重: %s", args.resume)

    set_asr_only_trainable(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("仅训练 ASR backend，可训练参数量: %s", f"{trainable:,}")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps, 1),
        eta_min=args.lr * 0.05,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始 ASR-only 训练，共 %d 个 epoch", args.epochs)
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            max_grad_norm=args.max_grad_norm,
        )
        logger.info("Epoch %d 平均 ASR loss: %.4f", epoch, avg_loss)
        save_checkpoint(
            checkpoint_dir / f"epoch_{epoch}.pt",
            epoch,
            model,
            optimizer,
            scaler,
            tokenizer,
            avg_loss,
        )
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            epoch,
            model,
            optimizer,
            scaler,
            tokenizer,
            avg_loss,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASR backend only on mixed command audio.")
    parser.add_argument("--jsonl_path", type=str, default="data/raw/test_set_a/labels.jsonl")
    parser.add_argument("--audio_dir", type=str, default="data/raw/test_set_a")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_asr_mixed")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
