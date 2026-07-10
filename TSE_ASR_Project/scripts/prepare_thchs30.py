#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download and prepare THCHS-30 manifests for synthetic TSE training."""

import argparse
import json
import random
import shutil
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


MAX_ARCHIVE_BYTES = 7 * 1024 ** 3


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def download(url: str, archive: Path, no_check_certificate: bool = False) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and archive.stat().st_size > 0:
        print(f"Archive exists: {archive} ({archive.stat().st_size / 1024 ** 3:.2f}G)")
        if archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise RuntimeError(f"Archive is larger than 7GB: {archive}")
        return
    downloader = shutil.which("aria2c")
    if downloader:
        run([
            "aria2c",
            "-x", "8",
            "-s", "8",
            "-c",
            "--file-allocation=none",
            "--check-certificate=false" if no_check_certificate else "--check-certificate=true",
            "-d", str(archive.parent),
            "-o", archive.name,
            url,
        ])
    else:
        downloader = shutil.which("wget")
    if downloader and Path(downloader).name == "wget":
        cmd = ["wget", "-c", "-O", str(archive)]
        if no_check_certificate:
            cmd.append("--no-check-certificate")
        cmd.append(url)
        run(cmd)
    elif not shutil.which("aria2c"):
        downloader = shutil.which("curl")
        if not downloader:
            raise RuntimeError("Neither wget nor curl is available.")
        run(["curl", "-L", "-C", "-", "-o", str(archive), url])
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise RuntimeError(f"Archive is larger than 7GB: {archive}")


def extract(archive: Path, root: Path) -> None:
    marker = root / ".extracted"
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


def read_transcript(trn_path: Path) -> str:
    lines = trn_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return ""
    return "".join(lines[0].strip().split())


def speaker_id_from_wav(wav_path: Path) -> str:
    stem = wav_path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem[:3]


def build_clean_manifest(root: Path, clean_manifest: Path) -> List[Dict[str, str]]:
    wavs = sorted(root.rglob("*.wav"))
    rows: List[Dict[str, str]] = []
    seen = set()
    for wav in wavs:
        trn = Path(str(wav) + ".trn")
        if not trn.is_file():
            continue
        text = read_transcript(trn)
        if not text:
            continue
        key = str(wav.resolve())
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "speaker": speaker_id_from_wav(wav),
                "wav": key,
                "text": text,
            }
        )
    clean_manifest.parent.mkdir(parents=True, exist_ok=True)
    with clean_manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def write_synthetic_manifests(
    rows: List[Dict[str, str]],
    train_manifest: Path,
    dev_manifest: Path,
    max_pairs: int,
    dev_speaker_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    by_speaker: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    speakers = sorted(spk for spk, items in by_speaker.items() if len(items) >= 2)
    if len(speakers) < 3:
        raise RuntimeError("Need at least 3 speakers with 2+ utterances.")

    rng.shuffle(speakers)
    dev_count = max(1, int(len(speakers) * dev_speaker_ratio))
    dev_speakers = set(speakers[:dev_count])
    train_speakers = set(speakers[dev_count:])

    def make_pairs(split_speakers: set, count: int) -> List[Dict[str, str]]:
        split = sorted(split_speakers)
        out: List[Dict[str, str]] = []
        for idx in range(count):
            is_target = idx % 4 != 0
            enroll_spk = rng.choice(split)
            if is_target:
                target_spk = enroll_spk
                interferer_spk = rng.choice([spk for spk in split if spk != target_spk])
            else:
                target_spk = rng.choice([spk for spk in split if spk != enroll_spk])
                interferer_spk = rng.choice([spk for spk in split if spk not in {enroll_spk, target_spk}])
            enroll, target = rng.sample(by_speaker[enroll_spk], 2) if is_target else (
                rng.choice(by_speaker[enroll_spk]),
                rng.choice(by_speaker[target_spk]),
            )
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

    train_rows = make_pairs(train_speakers, max_pairs)
    dev_rows = make_pairs(dev_speakers, max(200, max_pairs // 10))

    for path, items in [(train_manifest, train_rows), (dev_manifest, dev_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Clean utterances: {len(rows)}")
    print(f"Speakers train/dev: {len(train_speakers)}/{len(dev_speakers)}")
    print(f"Synthetic train: {train_manifest} ({len(train_rows)} rows)")
    print(f"Synthetic dev  : {dev_manifest} ({len(dev_rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare THCHS-30 for synthetic TSE training.")
    parser.add_argument("--url", default="https://openslr.magicdatatech.com/resources/18/data_thchs30.tgz")
    parser.add_argument("--root", default="/home/tzb/tjb/external_data/thchs30")
    parser.add_argument("--archive", default="/home/tzb/tjb/external_data/thchs30/data_thchs30.tgz")
    parser.add_argument("--clean_manifest", default="data/external/thchs30_clean.jsonl")
    parser.add_argument("--train_manifest", default="data/external/thchs30_synth_train.jsonl")
    parser.add_argument("--dev_manifest", default="data/external/thchs30_synth_dev.jsonl")
    parser.add_argument("--max_pairs", type=int, default=12000)
    parser.add_argument("--dev_speaker_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--no_check_certificate", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    archive = Path(args.archive)
    if not args.skip_download:
        download(args.url, archive, no_check_certificate=args.no_check_certificate)
    extract(archive, root)
    rows = build_clean_manifest(root, Path(args.clean_manifest))
    write_synthetic_manifests(
        rows,
        Path(args.train_manifest),
        Path(args.dev_manifest),
        args.max_pairs,
        args.dev_speaker_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()
