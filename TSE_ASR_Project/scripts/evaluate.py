#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained TSE-ASR checkpoint on a labeled dataset."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.train import SimpleCharTokenizer
from src.data.dataset import create_dataloader_from_config
from src.models.joint_model import JointTSEASR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, hyp_item in enumerate(hyp, start=1):
            cost = 0 if ref_item == hyp_item else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]


def restore_tokenizer(checkpoint: Dict[str, object]) -> SimpleCharTokenizer:
    tokenizer = SimpleCharTokenizer()
    char_to_id = checkpoint.get("tokenizer_char_to_id")
    if not isinstance(char_to_id, dict):
        raise RuntimeError("Checkpoint does not contain tokenizer_char_to_id.")

    tokenizer.char_to_id = {str(ch): int(idx) for ch, idx in char_to_id.items()}
    tokenizer.id_to_char = {idx: ch for ch, idx in tokenizer.char_to_id.items()}
    tokenizer.vocab_size = int(checkpoint.get("tokenizer_vocab_size", len(tokenizer.char_to_id) + 1))
    return tokenizer


def ctc_decode_batch(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    tokenizer: SimpleCharTokenizer,
) -> List[str]:
    pred_ids = logits.argmax(dim=-1).detach().cpu()
    lengths_cpu = lengths.detach().cpu().tolist()
    decoded: List[str] = []

    for row, length in zip(pred_ids, lengths_cpu):
        collapsed: List[int] = []
        prev = None
        for token_id in row[: int(length)].tolist():
            if token_id != prev and token_id != tokenizer.blank_id:
                collapsed.append(int(token_id))
            prev = token_id
        decoded.append(tokenizer.decode(collapsed))

    return decoded


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


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train first, or pass --checkpoint."
        )

    device = choose_device(args.device)
    logger.info("Using device: %s", device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = restore_tokenizer(checkpoint)

    model = build_model(tokenizer.vocab_size, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloader = create_dataloader_from_config(
        jsonl_path=args.jsonl_path,
        root_dir=args.audio_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sample_rate=16000,
    )

    total = 0
    reject_correct = 0
    pos_total = 0
    pos_accepted = 0
    pos_exact = 0
    neg_total = 0
    neg_rejected = 0
    cer_errors = 0
    cer_chars = 0

    for batch_idx, batch in enumerate(dataloader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        enroll_wavs = batch["enroll_wavs"].to(device, non_blocking=True)
        enroll_lengths = batch["enroll_lengths"].to(device, non_blocking=True)
        mixed_wavs = batch["mixed_wavs"].to(device, non_blocking=True)
        mixed_lengths = batch["mixed_lengths"].to(device, non_blocking=True)
        is_targets = batch["is_targets"].to(device, non_blocking=True)
        text_labels: Iterable[str] = batch["text_labels"]

        outputs = model(
            enroll_wavs=enroll_wavs,
            enroll_lengths=enroll_lengths,
            mixed_wavs=mixed_wavs,
            mixed_lengths=mixed_lengths,
        )

        reject_pred = outputs["reject_logits"].argmax(dim=-1)
        if args.asr_source == "mixed":
            asr_logits, asr_lengths = model.asr_backend(mixed_wavs, mixed_lengths)
        else:
            asr_logits, asr_lengths = outputs["asr_logits"], outputs["asr_lengths"]
        decoded = ctc_decode_batch(asr_logits, asr_lengths, tokenizer)

        labels = list(text_labels)
        for pred_class, target_class, pred_text, ref_text in zip(
            reject_pred.detach().cpu().tolist(),
            is_targets.detach().cpu().tolist(),
            decoded,
            labels,
        ):
            total += 1
            reject_correct += int(pred_class == target_class)

            if int(target_class) == 1:
                pos_total += 1
                pos_accepted += int(pred_class == 1)
                err = edit_distance(list(ref_text), list(pred_text))
                cer_errors += err
                cer_chars += max(len(ref_text), 1)
                pos_exact += int(pred_text == ref_text)
            else:
                neg_total += 1
                neg_rejected += int(pred_class == 0)

        if (batch_idx + 1) % args.log_interval == 0:
            logger.info("Processed %d/%d batches", batch_idx + 1, len(dataloader))

    cer = cer_errors / max(cer_chars, 1)
    rr = neg_rejected / max(neg_total, 1)
    reject_acc = reject_correct / max(total, 1)
    pos_accept_rate = pos_accepted / max(pos_total, 1)
    pos_exact_rate = pos_exact / max(pos_total, 1)

    print("\nEvaluation result")
    print("=================")
    print(f"checkpoint       : {checkpoint_path}")
    print(f"samples          : {total}")
    print(f"pos / neg        : {pos_total} / {neg_total}")
    print(f"CER on pos       : {cer:.4f} ({cer_errors}/{cer_chars})")
    print(f"RR on neg        : {rr:.4f} ({neg_rejected}/{neg_total})")
    print(f"reject accuracy  : {reject_acc:.4f} ({reject_correct}/{total})")
    print(f"pos accept rate  : {pos_accept_rate:.4f} ({pos_accepted}/{pos_total})")
    print(f"pos exact match  : {pos_exact_rate:.4f} ({pos_exact}/{pos_total})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TSE-ASR checkpoint.")
    parser.add_argument("--jsonl_path", type=str, default="data/raw/test_set_a/labels.jsonl")
    parser.add_argument("--audio_dir", type=str, default="data/raw/test_set_a")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--asr_source", type=str, default="clean", choices=["clean", "mixed"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
