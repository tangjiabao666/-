#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train TSE, speaker encoder, and rejection head with a frozen ASR backend."""

import argparse
import logging
import os
import random
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


def set_stage_trainable(
    model: JointTSEASR,
    train_speaker: bool,
    train_tse: bool,
    train_fusion: bool,
) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if train_tse:
        for param in model.tse_extractor.parameters():
            param.requires_grad = True
    if train_fusion:
        for param in model.fusion_gate.parameters():
            param.requires_grad = True
    for param in model.rejection_head.parameters():
        param.requires_grad = True
    if train_speaker:
        for param in model.speaker_encoder.parameters():
            param.requires_grad = True
    model.asr_backend.eval()


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    is_targets: torch.Tensor,
) -> torch.Tensor:
    if not bool(is_targets.bool().any()):
        return pred.new_tensor(0.0)
    max_len = pred.shape[-1]
    mask = torch.arange(max_len, device=pred.device).view(1, 1, -1) < lengths.view(-1, 1, 1)
    mask = mask & is_targets.bool().view(-1, 1, 1)
    diff = (pred - target).abs() * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff.sum() / denom


def save_checkpoint(
    path: Path,
    epoch: int,
    model: JointTSEASR,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    tokenizer: SimpleCharTokenizer,
    avg_losses: Dict[str, float],
    training_args: Dict[str, object],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_char_to_id": tokenizer.char_to_id,
            "avg_losses": avg_losses,
            "training_mode": "tse_stage_frozen_asr",
            "training_args": training_args,
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
    asr_weight: float,
    reject_weight: float,
    signal_weight: float,
    neg_class_weight: float,
    pos_class_weight: float,
) -> Dict[str, float]:
    model.train()
    for module in [
        model.speaker_encoder,
        model.tse_extractor,
        model.fusion_gate,
        model.asr_backend,
    ]:
        if not any(param.requires_grad for param in module.parameters()):
            module.eval()
    model.asr_backend.eval()
    ctc_loss = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True, reduction="none")
    ce_weights = torch.tensor(
        [neg_class_weight, pos_class_weight],
        dtype=torch.float32,
        device=device,
    )
    ce_loss = torch.nn.CrossEntropyLoss(weight=ce_weights)
    sums = {"total": 0.0, "asr": 0.0, "reject": 0.0, "signal": 0.0}
    steps = 0

    for batch_idx, batch in enumerate(dataloader):
        enroll_wavs = batch["enroll_wavs"].to(device, non_blocking=True)
        enroll_lengths = batch["enroll_lengths"].to(device, non_blocking=True)
        mixed_wavs = batch["mixed_wavs"].to(device, non_blocking=True)
        mixed_lengths = batch["mixed_lengths"].to(device, non_blocking=True)
        text_labels: List[str] = batch["text_labels"]
        is_targets = batch["is_targets"].to(device, non_blocking=True).long()

        max_len = int(mixed_lengths.max().item())
        mixed_wavs = mixed_wavs[..., :max_len]

        text_targets, text_target_lengths = collate_text_batch(text_labels, tokenizer, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(
                enroll_wavs=enroll_wavs,
                enroll_lengths=enroll_lengths,
                mixed_wavs=mixed_wavs,
                mixed_lengths=mixed_lengths,
            )
            log_probs = F.log_softmax(outputs["asr_logits"], dim=-1).transpose(0, 1)
            asr_each = ctc_loss(
                log_probs,
                text_targets,
                outputs["asr_lengths"],
                text_target_lengths,
            )
            pos_mask = is_targets.float()
            num_pos = pos_mask.sum().clamp_min(1.0)
            asr_loss = (asr_each * pos_mask).sum() / num_pos
            reject_loss = ce_loss(outputs["reject_logits"], is_targets)
            signal_loss = masked_l1_loss(
                outputs["clean_wavs"],
                mixed_wavs,
                mixed_lengths,
                is_targets,
            )
            loss = asr_weight * asr_loss + reject_weight * reject_loss + signal_weight * signal_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        nn_utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        values = {
            "total": float(loss.detach().item()),
            "asr": float(asr_loss.detach().item()),
            "reject": float(reject_loss.detach().item()),
            "signal": float(signal_loss.detach().item()),
        }
        for key, value in values.items():
            sums[key] += value
        steps += 1

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                "Epoch %d | Batch %d/%d | Total=%.4f | ASR=%.4f | Rej=%.4f | Sig=%.5f | pos=%d | LR=%.2e",
                epoch,
                batch_idx + 1,
                len(dataloader),
                values["total"],
                values["asr"],
                values["reject"],
                values["signal"],
                int(is_targets.sum().item()),
                optimizer.param_groups[0]["lr"],
            )

    return {key: value / max(steps, 1) for key, value in sums.items()}


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    if not args.resume or not os.path.isfile(args.resume):
        raise FileNotFoundError("--resume must point to the ASR-trained checkpoint")
    checkpoint = torch.load(args.resume, map_location=device)
    tokenizer = restore_tokenizer(checkpoint)
    if tokenizer is None:
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
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    logger.info("已加载模型权重: %s", args.resume)
    if load_result.missing_keys:
        logger.info("新初始化参数: %s", ", ".join(load_result.missing_keys))
    if load_result.unexpected_keys:
        logger.info("checkpoint 中未使用参数: %s", ", ".join(load_result.unexpected_keys))

    set_stage_trainable(
        model,
        train_speaker=not args.freeze_speaker,
        train_tse=not args.freeze_tse,
        train_fusion=not args.freeze_fusion,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("冻结 ASR backend，可训练参数量: %s", f"{trainable:,}")

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

    logger.info("开始 TSE-stage 训练，共 %d 个 epoch", args.epochs)
    for epoch in range(1, args.epochs + 1):
        avg_losses = train_one_epoch(
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
            asr_weight=args.asr_weight,
            reject_weight=args.reject_weight,
            signal_weight=args.signal_weight,
            neg_class_weight=args.neg_class_weight,
            pos_class_weight=args.pos_class_weight,
        )
        logger.info(
            "Epoch %d 平均损失: Total=%.4f | ASR=%.4f | Rej=%.4f | Sig=%.5f",
            epoch,
            avg_losses["total"],
            avg_losses["asr"],
            avg_losses["reject"],
            avg_losses["signal"],
        )
        save_checkpoint(
            checkpoint_dir / f"epoch_{epoch}.pt",
            epoch,
            model,
            optimizer,
            scaler,
            tokenizer,
            avg_losses,
            vars(args),
        )
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            epoch,
            model,
            optimizer,
            scaler,
            tokenizer,
            avg_losses,
            vars(args),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TSE/rejection with frozen ASR feedback.")
    parser.add_argument("--jsonl_path", type=str, default="data/raw/test_set_a/labels.jsonl")
    parser.add_argument("--audio_dir", type=str, default="data/raw/test_set_a")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_tse_stage")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--asr_weight", type=float, default=1.0)
    parser.add_argument("--reject_weight", type=float, default=0.5)
    parser.add_argument("--signal_weight", type=float, default=0.02)
    parser.add_argument("--neg_class_weight", type=float, default=1.0)
    parser.add_argument("--pos_class_weight", type=float, default=1.0)
    parser.add_argument("--freeze_speaker", action="store_true")
    parser.add_argument("--freeze_tse", action="store_true")
    parser.add_argument("--freeze_fusion", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
