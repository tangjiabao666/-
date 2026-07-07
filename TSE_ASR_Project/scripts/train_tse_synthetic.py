#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train TSE/fusion/rejection with synthetic mixtures built from clean speech."""

import argparse
import json
import logging
import os
import random
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torchaudio
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.train import SimpleCharTokenizer, build_tokenizer_from_dataset  # noqa: E402
from src.models.joint_model import JointTSEASR  # noqa: E402
from src.utils.loss import si_snr  # noqa: E402


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


def load_pcm_wav(path: str) -> Tuple[torch.Tensor, int]:
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"Only 16-bit PCM WAV is supported by fallback: {path}")
    wav = torch.frombuffer(frames, dtype=torch.int16).float() / 32768.0
    if channels > 1:
        wav = wav.view(-1, channels).mean(dim=1)
    return wav.unsqueeze(0).clone(), sample_rate


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        wav, sr = load_pcm_wav(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def fit_length(wav: torch.Tensor, length: int, rng: random.Random) -> torch.Tensor:
    cur = wav.shape[-1]
    if cur == length:
        return wav
    if cur > length:
        start = rng.randint(0, cur - length)
        return wav[..., start:start + length]
    reps = (length + cur - 1) // max(cur, 1)
    return wav.repeat(1, reps)[..., :length]


def rms(wav: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(wav ** 2) + 1e-8)


class SyntheticTSEDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        sample_rate: int,
        snr_min_db: float,
        snr_max_db: float,
        max_seconds: float,
        seed: int,
    ) -> None:
        self.rows: List[Dict[str, object]] = []
        with open(manifest, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"No rows found in {manifest}")
        self.sample_rate = sample_rate
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.max_len = int(max_seconds * sample_rate) if max_seconds > 0 else None
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        rng = random.Random(self.seed + idx)
        enroll = load_audio(str(row["enroll_wav"]), self.sample_rate)
        target = load_audio(str(row["target_wav"]), self.sample_rate)
        interferer = load_audio(str(row["interferer_wav"]), self.sample_rate)

        if self.max_len is not None and target.shape[-1] > self.max_len:
            target = fit_length(target, self.max_len, rng)
        length = target.shape[-1]
        interferer = fit_length(interferer, length, rng)

        is_target = int(row["is_target"])
        snr_db = rng.uniform(self.snr_min_db, self.snr_max_db)
        if is_target:
            clean = target
            scale = rms(clean) / (rms(interferer) * (10.0 ** (snr_db / 20.0)) + 1e-8)
            mixed = clean + interferer * scale
        else:
            clean = torch.zeros_like(target)
            mixed = target

        peak = mixed.abs().max().clamp_min(1.0)
        mixed = mixed / peak
        clean = clean / peak
        return {
            "enroll_wav": enroll,
            "mixed_wav": mixed,
            "clean_wav": clean,
            "is_target": is_target,
            "text": str(row.get("text", "")),
        }


def pad_1d(items: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([item.shape[-1] for item in items], dtype=torch.long)
    max_len = int(lengths.max().item())
    out = items[0].new_zeros(len(items), 1, max_len)
    for idx, item in enumerate(items):
        out[idx, :, : item.shape[-1]] = item
    return out, lengths


def collate_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    enroll_wavs, enroll_lengths = pad_1d([item["enroll_wav"] for item in batch])
    mixed_wavs, mixed_lengths = pad_1d([item["mixed_wav"] for item in batch])
    clean_wavs, _ = pad_1d([item["clean_wav"] for item in batch])
    return {
        "enroll_wavs": enroll_wavs,
        "enroll_lengths": enroll_lengths,
        "mixed_wavs": mixed_wavs,
        "mixed_lengths": mixed_lengths,
        "clean_wavs": clean_wavs,
        "is_targets": torch.tensor([int(item["is_target"]) for item in batch], dtype=torch.long),
        "texts": [str(item["text"]) for item in batch],
    }


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


def set_trainable(model: JointTSEASR, train_speaker: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in [model.tse_extractor, model.fusion_gate, model.rejection_head]:
        for param in module.parameters():
            param.requires_grad = True
    if train_speaker:
        for param in model.speaker_encoder.parameters():
            param.requires_grad = True
    model.asr_backend.eval()


def masked_sisnr_loss(pred: torch.Tensor, target: torch.Tensor, is_targets: torch.Tensor) -> torch.Tensor:
    mask = is_targets.bool()
    if not bool(mask.any()):
        return pred.new_tensor(0.0)
    pred_pos = pred[mask].squeeze(1)
    target_pos = target[mask].squeeze(1)
    return -si_snr(pred_pos, target_pos).mean()


def save_checkpoint(
    path: Path,
    epoch: int,
    model: JointTSEASR,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    tokenizer: SimpleCharTokenizer,
    avg_losses: Dict[str, float],
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
            "training_mode": "synthetic_clean_tse",
        },
        path,
    )


def train_one_epoch(
    model: JointTSEASR,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    model.asr_backend.eval()
    ce_weights = torch.tensor([args.neg_class_weight, args.pos_class_weight], device=device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=ce_weights)
    sums = {"total": 0.0, "tse": 0.0, "fused": 0.0, "reject": 0.0}
    steps = 0

    for batch_idx, batch in enumerate(dataloader):
        enroll_wavs = batch["enroll_wavs"].to(device, non_blocking=True)
        enroll_lengths = batch["enroll_lengths"].to(device, non_blocking=True)
        mixed_wavs = batch["mixed_wavs"].to(device, non_blocking=True)
        mixed_lengths = batch["mixed_lengths"].to(device, non_blocking=True)
        clean_wavs = batch["clean_wavs"].to(device, non_blocking=True)
        is_targets = batch["is_targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(enroll_wavs, enroll_lengths, mixed_wavs, mixed_lengths)
            tse_loss = masked_sisnr_loss(outputs["tse_wavs"], clean_wavs, is_targets)
            fused_loss = masked_sisnr_loss(outputs["clean_wavs"], clean_wavs, is_targets)
            reject_loss = ce_loss(outputs["reject_logits"], is_targets)
            loss = args.tse_weight * tse_loss + args.fused_weight * fused_loss + args.reject_weight * reject_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable = [p for p in model.parameters() if p.requires_grad]
        nn_utils.clip_grad_norm_(trainable, args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        values = {
            "total": float(loss.detach().item()),
            "tse": float(tse_loss.detach().item()),
            "fused": float(fused_loss.detach().item()),
            "reject": float(reject_loss.detach().item()),
        }
        for key, value in values.items():
            sums[key] += value
        steps += 1
        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(
                "Epoch %d | Batch %d/%d | Total=%.4f | TSE=%.4f | Fused=%.4f | Rej=%.4f | pos=%d | LR=%.2e",
                epoch,
                batch_idx + 1,
                len(dataloader),
                values["total"],
                values["tse"],
                values["fused"],
                values["reject"],
                int(is_targets.sum().item()),
                optimizer.param_groups[0]["lr"],
            )

    return {key: value / max(steps, 1) for key, value in sums.items()}


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    if not os.path.isfile(args.resume):
        raise FileNotFoundError(args.resume)
    checkpoint = torch.load(args.resume, map_location=device)
    tokenizer = restore_tokenizer(checkpoint)
    if tokenizer is None:
        tokenizer = build_tokenizer_from_dataset(args.jsonl_path, args.audio_dir)

    dataset = SyntheticTSEDataset(
        manifest=args.manifest,
        sample_rate=args.sample_rate,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        max_seconds=args.max_seconds,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(tokenizer.vocab_size, device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    logger.info("已加载模型权重: %s", args.resume)
    if load_result.missing_keys:
        logger.info("新初始化参数: %s", ", ".join(load_result.missing_keys))
    if load_result.unexpected_keys:
        logger.info("checkpoint 中未使用参数: %s", ", ".join(load_result.unexpected_keys))
    set_trainable(model, train_speaker=not args.freeze_speaker)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("合成 TSE 训练，可训练参数量: %s", f"{trainable_count:,}")

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

    for epoch in range(1, args.epochs + 1):
        avg_losses = train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch, args)
        logger.info(
            "Epoch %d 平均损失: Total=%.4f | TSE=%.4f | Fused=%.4f | Rej=%.4f",
            epoch,
            avg_losses["total"],
            avg_losses["tse"],
            avg_losses["fused"],
            avg_losses["reject"],
        )
        save_checkpoint(checkpoint_dir / f"epoch_{epoch}.pt", epoch, model, optimizer, scaler, tokenizer, avg_losses)
        save_checkpoint(checkpoint_dir / "latest.pt", epoch, model, optimizer, scaler, tokenizer, avg_losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic clean-target TSE training.")
    parser.add_argument("--manifest", default="data/external/thchs30_synth_train.jsonl")
    parser.add_argument("--resume", default="checkpoints_reject_calibrated/latest.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints_tse_synthetic")
    parser.add_argument("--jsonl_path", default="data/raw/test_set_a/labels.jsonl")
    parser.add_argument("--audio_dir", default="data/raw/test_set_a")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--snr_min_db", type=float, default=-3.0)
    parser.add_argument("--snr_max_db", type=float, default=6.0)
    parser.add_argument("--max_seconds", type=float, default=6.0)
    parser.add_argument("--tse_weight", type=float, default=1.0)
    parser.add_argument("--fused_weight", type=float, default=0.5)
    parser.add_argument("--reject_weight", type=float, default=0.5)
    parser.add_argument("--neg_class_weight", type=float, default=1.5)
    parser.add_argument("--pos_class_weight", type=float, default=1.0)
    parser.add_argument("--freeze_speaker", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
