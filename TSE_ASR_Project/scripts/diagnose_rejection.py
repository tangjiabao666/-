#!/usr/bin/env python3
"""Export sample-level rejection diagnostics for one or more checkpoints."""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.evaluate import build_model, choose_device, restore_tokenizer
from src.data.dataset import create_dataloader_from_config


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    return numerator / math.sqrt(dx * dy) if dx > 0 and dy > 0 else float("nan")


def auc(rows: Sequence[Dict[str, float]]) -> float:
    pos = [r["margin"] for r in rows if r["target"] == 1]
    neg = [r["margin"] for r in rows if r["target"] == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def class_summary(rows: Sequence[Dict[str, float]], target: int) -> Dict[str, float]:
    selected = [r for r in rows if r["target"] == target]
    margins = [r["margin"] for r in selected]
    return {
        "count": len(selected),
        "margin_mean": statistics.fmean(margins),
        "margin_p10": quantile(margins, 0.10),
        "margin_p50": quantile(margins, 0.50),
        "margin_p90": quantile(margins, 0.90),
        "duration_mean_s": statistics.fmean(r["duration_s"] for r in selected),
        "mixed_rms_mean": statistics.fmean(r["mixed_rms"] for r in selected),
        "tse_energy_ratio_mean": statistics.fmean(r["tse_energy_ratio"] for r in selected),
        "fusion_gate_mean": statistics.fmean(r["fusion_gate"] for r in selected),
        "enroll_mixed_cos_mean": statistics.fmean(r["enroll_mixed_cos"] for r in selected),
        "enroll_tse_cos_mean": statistics.fmean(r["enroll_tse_cos"] for r in selected),
    }


def threshold_metrics(rows: Sequence[Dict[str, float]], threshold: float) -> Dict[str, float]:
    pos = [r for r in rows if r["target"] == 1]
    neg = [r for r in rows if r["target"] == 0]
    pos_accept = sum(r["margin"] >= threshold for r in pos) / len(pos)
    rr = sum(r["margin"] < threshold for r in neg) / len(neg)
    accuracy = sum((r["margin"] >= threshold) == bool(r["target"]) for r in rows) / len(rows)
    return {"threshold": threshold, "pos_accept": pos_accept, "rr": rr, "accuracy": accuracy}


def diagnose(checkpoint_path: Path, args: argparse.Namespace, records: List[dict]) -> Dict[str, object]:
    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = restore_tokenizer(checkpoint)
    model = build_model(tokenizer.vocab_size, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    loader = create_dataloader_from_config(
        jsonl_path=args.jsonl_path,
        root_dir=args.audio_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        sample_rate=16000,
    )

    rows: List[Dict[str, object]] = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            enroll = batch["enroll_wavs"].to(device)
            enroll_len = batch["enroll_lengths"].to(device)
            mixed = batch["mixed_wavs"].to(device)
            mixed_len = batch["mixed_lengths"].to(device)
            target = batch["is_targets"].to(device)
            outputs = model(enroll, enroll_len, mixed, mixed_len)
            margin = outputs["reject_logits"][:, 1] - outputs["reject_logits"][:, 0]
            enroll_emb = model.speaker_encoder(enroll, enroll_len)
            mixed_emb = model.speaker_encoder(mixed, mixed_len)
            tse_emb = model.speaker_encoder(outputs["tse_wavs"], mixed_len)
            mixed_cos = F.cosine_similarity(enroll_emb, mixed_emb)
            tse_cos = F.cosine_similarity(enroll_emb, tse_emb)

            for i in range(enroll.shape[0]):
                n = int(mixed_len[i].item())
                e = int(enroll_len[i].item())
                mixed_rms = mixed[i, 0, :n].square().mean().sqrt().item()
                enroll_rms = enroll[i, 0, :e].square().mean().sqrt().item()
                tse_rms = outputs["tse_wavs"][i, 0, :n].square().mean().sqrt().item()
                clean_rms = outputs["clean_wavs"][i, 0, :n].square().mean().sqrt().item()
                source = records[offset + i] if offset + i < len(records) else {}
                rows.append({
                    "index": offset + i,
                    "id": source.get("id", offset + i),
                    "mixed_path": source.get("识别音频", ""),
                    "target": int(target[i].item()),
                    "margin": margin[i].item(),
                    "duration_s": n / 16000.0,
                    "enroll_duration_s": e / 16000.0,
                    "mixed_rms": mixed_rms,
                    "enroll_rms": enroll_rms,
                    "tse_rms": tse_rms,
                    "clean_rms": clean_rms,
                    "tse_energy_ratio": tse_rms / max(mixed_rms, 1e-8),
                    "fusion_gate": outputs["fusion_gate"][i].item(),
                    "enroll_mixed_cos": mixed_cos[i].item(),
                    "enroll_tse_cos": tse_cos[i].item(),
                })
            offset += enroll.shape[0]

    checkpoint_name = checkpoint_path.parent.name + "_" + checkpoint_path.stem
    csv_path = args.output_dir / f"{checkpoint_name}_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    thresholds = [threshold_metrics(rows, t / 20.0) for t in range(-40, 61)]
    best_balanced = max(thresholds, key=lambda x: 0.5 * (x["pos_accept"] + x["rr"]))
    best_pos80 = max((x for x in thresholds if x["pos_accept"] >= 0.8), key=lambda x: x["rr"])
    correlations = {
        key: pearson([float(r["margin"]) for r in rows], [float(r[key]) for r in rows])
        for key in ["duration_s", "mixed_rms", "tse_energy_ratio", "fusion_gate", "enroll_mixed_cos", "enroll_tse_cos"]
    }
    false_accepts = sorted(
        [r for r in rows if r["target"] == 0 and r["margin"] >= 0],
        key=lambda r: r["margin"],
        reverse=True,
    )[:15]
    summary = {
        "checkpoint": str(checkpoint_path),
        "sample_csv": str(csv_path),
        "auc": auc(rows),
        "positive": class_summary(rows, 1),
        "negative": class_summary(rows, 0),
        "threshold_0": threshold_metrics(rows, 0.0),
        "best_balanced": best_balanced,
        "best_rr_at_pos80": best_pos80,
        "margin_correlations_all": correlations,
        "top_false_accepts_at_threshold_0": false_accepts,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", default="data/splits/dev.jsonl")
    parser.add_argument("--audio_dir", default="data/raw/test_set_a")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("diagnostics/rejection"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.jsonl_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    summaries = [diagnose(Path(path), args, records) for path in args.checkpoints]
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
