#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a small LibriSpeech split for external clean-target TSE training."""

import argparse
import json
import random
import shutil
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def download(url: str, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and archive.stat().st_size > 0 and not Path(str(archive) + ".aria2").exists():
        print(f"Archive exists: {archive} ({archive.stat().st_size / 1024 ** 2:.1f}MB)")
        return
    if shutil.which("aria2c"):
        run([
            "aria2c",
            "-x", "8",
            "-s", "8",
            "-c",
            "--file-allocation=none",
            "--check-certificate=false",
            "-d", str(archive.parent),
            "-o", archive.name,
            url,
        ])
    elif shutil.which("wget"):
        run(["wget", "-c", "--no-check-certificate", "-O", str(archive), url])
    else:
        run(["curl", "-L", "-C", "-", "-o", str(archive), url])


def extract(archive: Path, root: Path) -> None:
    marker = root / ".librispeech_dev_clean_extracted"
    if marker.exists():
        print(f"Already extracted: {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        root_resolved = root.resolve()
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if not str(target).startswith(str(root_resolved)):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tar.extractall(root)
    marker.write_text("ok\n", encoding="utf-8")


def collect_rows(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for trans_path in sorted(root.rglob("*.trans.txt")):
        trans: Dict[str, str] = {}
        for line in trans_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                trans[parts[0]] = parts[1]
        for utt_id, text in trans.items():
            wav = trans_path.parent / f"{utt_id}.flac"
            if wav.is_file():
                speaker = utt_id.split("-", 1)[0]
                rows.append({"speaker": speaker, "wav": str(wav.resolve()), "text": text})
    if not rows:
        raise RuntimeError(f"No LibriSpeech rows found under {root}")
    return rows


def write_pairs(rows: List[Dict[str, str]], train_manifest: Path, dev_manifest: Path, max_pairs: int, seed: int) -> None:
    rng = random.Random(seed)
    by_speaker: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    speakers = sorted(spk for spk, items in by_speaker.items() if len(items) >= 2)
    rng.shuffle(speakers)
    dev_count = max(1, len(speakers) // 5)
    dev_speakers = set(speakers[:dev_count])
    train_speakers = set(speakers[dev_count:])

    def make(split_speakers: set, count: int) -> List[Dict[str, object]]:
        split = sorted(split_speakers)
        out: List[Dict[str, object]] = []
        for idx in range(count):
            is_target = idx % 4 != 0
            enroll_spk = rng.choice(split)
            if is_target:
                target_spk = enroll_spk
                interferer_spk = rng.choice([spk for spk in split if spk != target_spk])
                enroll, target = rng.sample(by_speaker[enroll_spk], 2)
            else:
                target_spk = rng.choice([spk for spk in split if spk != enroll_spk])
                interferer_spk = rng.choice([spk for spk in split if spk not in {enroll_spk, target_spk}])
                enroll = rng.choice(by_speaker[enroll_spk])
                target = rng.choice(by_speaker[target_spk])
            interferer = rng.choice(by_speaker[interferer_spk])
            out.append(
                {
                    "id": idx,
                    "is_target": int(is_target),
                    "speaker": enroll_spk,
                    "enroll_wav": enroll["wav"],
                    "target_wav": target["wav"],
                    "interferer_wav": interferer["wav"],
                    "text": target["text"] if is_target else "",
                }
            )
        return out

    train_rows = make(train_speakers, max_pairs)
    dev_rows = make(dev_speakers, max(200, max_pairs // 10))
    for path, items in [(train_manifest, train_rows), (dev_manifest, dev_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"utterances: {len(rows)}")
    print(f"speakers train/dev: {len(train_speakers)}/{len(dev_speakers)}")
    print(f"train pairs: {len(train_rows)} -> {train_manifest}")
    print(f"dev pairs: {len(dev_rows)} -> {dev_manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech dev-clean synthetic TSE manifests.")
    parser.add_argument("--url", default="https://openslr.magicdatatech.com/resources/12/dev-clean.tar.gz")
    parser.add_argument("--root", default="/home/tzb/tjb/external_data/librispeech")
    parser.add_argument("--archive", default="/home/tzb/tjb/external_data/librispeech/dev-clean.tar.gz")
    parser.add_argument("--train_manifest", default="data/external/librispeech_synth_train.jsonl")
    parser.add_argument("--dev_manifest", default="data/external/librispeech_synth_dev.jsonl")
    parser.add_argument("--max_pairs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    archive = Path(args.archive)
    root = Path(args.root)
    download(args.url, archive)
    extract(archive, root)
    rows = collect_rows(root)
    write_pairs(rows, Path(args.train_manifest), Path(args.dev_manifest), args.max_pairs, args.seed)


if __name__ == "__main__":
    main()
