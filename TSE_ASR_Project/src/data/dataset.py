#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSE-ASR 项目 —— 复杂交互场景下的目标说话人语音识别数据集加载模块。

本模块提供了一个 PyTorch Dataset 类 ``ComplexSpeechDataset``，用于解析
组委会提供的 JSONL 标注文件，并加载对应的唤醒音频（enrollment）与识别音频
（mixed / command），供后续 TSE-ASR 联合模型训练与评估使用。

数据格式说明
------------
数据集分为两组：
- pos（正样本）: ``识别文本`` 字段非空，用于 CER 评估。
- neg（拒识样本）: ``识别文本`` 字段为 ``null``，用于 RR 评估。

每条 JSONL 记录包含以下字段（中文字段名）::

    {
        "id": <int>,
        "唤醒音频": "<相对路径>",
        "唤醒文本": "<唤醒词文本>",
        "识别音频": "<相对路径>",
        "识别文本": "<识别结果文本 | null>"
    }

音频文件存放在 ``{root_dir}/pos/`` 或 ``{root_dir}/neg/`` 下，
路径字段中已包含 ``pos/`` 或 ``neg/`` 前缀。
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

# ---------- 日志配置 ----------
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)


class ComplexSpeechDataset(Dataset):
    """复杂交互场景数据集。

    读取 JSONL 标注文件，为每条交互记录加载一对唤醒音频与识别音频，
    并提供是否为拒识样本 (is_target) 的标识。

    Parameters
    ----------
    jsonl_path : str
        JSONL 标注文件的路径。
    root_dir : str
        音频文件根目录（例如 ``data/raw/test_set_a/``）。
    sample_rate : int, optional
        目标采样率，默认 16000。
    """

    def __init__(
        self,
        jsonl_path: str,
        root_dir: str,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate

        # ---------- 解析 JSONL，仅保留能被找到的条目 ----------
        self._samples: List[Dict[str, Any]] = []
        self._parse_jsonl(jsonl_path)

        if len(self._samples) == 0:
            raise RuntimeError(
                f"未能从 '{jsonl_path}' 中加载到任何有效样本，"
                f"请检查 JSONL 内容和 root_dir='{root_dir}' 是否匹配。"
            )

        logger.info("ComplexSpeechDataset 初始化完成，共加载 %d 条样本。", len(self._samples))

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _parse_jsonl(self, jsonl_path: str) -> None:
        """解析 JSONL 文件，过滤掉音频文件不存在的条目。

        Parameters
        ----------
        jsonl_path : str
            JSONL 文件路径。
        """
        skipped_count = 0

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("第 %d 行 JSON 解析失败: %s", line_no, exc)
                    skipped_count += 1
                    continue

                # 提取必要字段（使用数据集实际的中文字段名）
                enroll_path_raw: Optional[str] = record.get("唤醒音频")
                mixed_path_raw: Optional[str] = record.get("识别音频")
                enroll_text: Optional[str] = record.get("唤醒文本")
                text_label: Optional[str] = record.get("识别文本")

                # 校验必须字段是否存在
                if enroll_path_raw is None or mixed_path_raw is None:
                    logger.warning(
                        "第 %d 行缺少 '唤醒音频' 或 '识别音频' 字段，已跳过。",
                        line_no,
                    )
                    skipped_count += 1
                    continue

                # 拼接完整路径
                enroll_path = self.root_dir / enroll_path_raw
                mixed_path = self.root_dir / mixed_path_raw

                # 检查音频文件是否存在
                if not enroll_path.is_file():
                    logger.warning(
                        "唤醒音频文件不存在: %s (第 %d 行)，已跳过。",
                        enroll_path,
                        line_no,
                    )
                    skipped_count += 1
                    continue
                if not mixed_path.is_file():
                    logger.warning(
                        "识别音频文件不存在: %s (第 %d 行)，已跳过。",
                        mixed_path,
                        line_no,
                    )
                    skipped_count += 1
                    continue

                # 判断是否为拒识样本：识别文本为 null 或空字符串即为 neg
                is_target: bool = text_label is not None and len(str(text_label).strip()) > 0

                sample: Dict[str, Any] = {
                    "enroll_path": str(enroll_path),
                    "mixed_path": str(mixed_path),
                    "enroll_text": enroll_text or "",
                    "text_label": str(text_label) if is_target else "",
                    "is_target": int(is_target),  # 1 = pos, 0 = neg
                }
                self._samples.append(sample)

        if skipped_count > 0:
            logger.warning(
                "JSONL 解析完成: 共跳过 %d 条无效样本（缺少字段或音频缺失）。",
                skipped_count,
            )

    # ------------------------------------------------------------------
    # PyTorch Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """返回数据集中的样本总数。"""
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """返回指定索引的样本数据。

        针对可能出现的音频加载异常（例如文件损坏、权限不足等），
        采用“重试 + 随机替换”策略：在当前索引加载失败时，
        随机选取另一条有效样本返回，避免训练中断。

        Parameters
        ----------
        index : int
            样本索引。

        Returns
        -------
        dict
            包含以下键的字典:
            - ``enroll_wav`` (torch.Tensor): 唤醒音频波形，shape [1, T]
            - ``mixed_wav`` (torch.Tensor): 识别音频波形，shape [1, T]
            - ``text_label`` (str): 识别文本标签
            - ``is_target`` (int): 1 表示正样本 (pos)，0 表示拒识 (neg)
            - ``enroll_text`` (str): 唤醒词文本
        """
        # 最多尝试 5 次加载
        max_retries = 5
        current_index = index

        for attempt in range(max_retries):
            sample = self._samples[current_index]

            try:
                # 加载唤醒音频 (enrollment)
                enroll_wav, enroll_sr = torchaudio.load(sample["enroll_path"])
                enroll_wav = self._resample_if_needed(enroll_wav, enroll_sr)

                # 加载识别音频 (mixed / command)
                mixed_wav, mixed_sr = torchaudio.load(sample["mixed_path"])
                mixed_wav = self._resample_if_needed(mixed_wav, mixed_sr)

                return {
                    "enroll_wav": enroll_wav,
                    "mixed_wav": mixed_wav,
                    "text_label": sample["text_label"],
                    "is_target": sample["is_target"],
                    "enroll_text": sample["enroll_text"],
                }

            except Exception as exc:
                logger.warning(
                    "加载音频失败 (索引=%d, 尝试=%d/%d): %s — %s",
                    current_index,
                    attempt + 1,
                    max_retries,
                    sample["mixed_path"],
                    exc,
                )
                # 第一次失败时保持当前索引重试，之后随机替换
                if attempt == 0:
                    current_index = index
                else:
                    current_index = random.randint(0, len(self._samples) - 1)

        # 理论上不会执行到这里，因为重试策略会一直替换索引
        raise RuntimeError(
            f"多次尝试后仍无法加载音频，初始索引为 {index}，"
            f"请检查音频文件是否损坏。"
        )

    def _resample_if_needed(
        self,
        waveform: torch.Tensor,
        orig_sr: int,
    ) -> torch.Tensor:
        """将音频重采样到目标采样率，并确保为单声道。

        Parameters
        ----------
        waveform : torch.Tensor
            原始波形张量，shape [C, T]。
        orig_sr : int
            原始采样率。

        Returns
        -------
        torch.Tensor
            重采样后的波形张量，shape [1, T]。
        """
        # 声道处理：多于 1 声道则取平均转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 采样率转换
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate,
            )
            waveform = resampler(waveform)

        return waveform

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, int]:
        """返回数据集的基本统计信息。

        Returns
        -------
        dict
            包含 ``total``, ``pos``, ``neg`` 计数的字典。
        """
        pos_count = sum(1 for s in self._samples if s["is_target"] == 1)
        neg_count = len(self._samples) - pos_count
        return {
            "total": len(self._samples),
            "pos": pos_count,
            "neg": neg_count,
        }


# ====================================================================
# 模块级便捷函数
# ====================================================================

def create_dataloader_from_config(
    jsonl_path: str,
    root_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    sample_rate: int = 16000,
) -> torch.utils.data.DataLoader:
    """根据配置参数快速创建 DataLoader 的便捷函数。

    Parameters
    ----------
    jsonl_path : str
        JSONL 标注文件路径。
    root_dir : str
        音频文件根目录。
    batch_size : int, optional
        批次大小，默认 8。
    shuffle : bool, optional
        是否打乱数据，默认 True。
    num_workers : int, optional
        数据加载子进程数，默认 4。
    sample_rate : int, optional
        目标采样率，默认 16000。

    Returns
    -------
    torch.utils.data.DataLoader
        配置好的 DataLoader 实例。
    """
    dataset = ComplexSpeechDataset(
        jsonl_path=jsonl_path,
        root_dir=root_dir,
        sample_rate=sample_rate,
    )

    # 输出统计信息
    stats = dataset.get_statistics()
    logger.info(
        "创建 DataLoader: pos=%d, neg=%d, total=%d",
        stats["pos"],
        stats["neg"],
        stats["total"],
    )

    # collate_fn: 将不等长的音频 padding 成对齐的张量，并记录真实长度，
    # 这样模型才知道哪里是有效语音，哪里是补上去的静音（padding）。
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """进阶批次组装：将不等长的音频 padding 成对齐的张量，并记录真实长度。"""
        from torch.nn.utils.rnn import pad_sequence

        # 提取波形，注意要把 [1, T] 挤压成 [T] 才能用 pad_sequence
        batch_enroll = [item["enroll_wav"].squeeze(0) for item in batch]
        batch_mixed = [item["mixed_wav"].squeeze(0) for item in batch]

        # 记录原始长度
        enroll_lengths = torch.tensor([wav.shape[0] for wav in batch_enroll], dtype=torch.long)
        mixed_lengths = torch.tensor([wav.shape[0] for wav in batch_mixed], dtype=torch.long)

        # Padding 操作，batch_first=True 保证输出 shape 为 [B, T]
        enroll_padded = pad_sequence(batch_enroll, batch_first=True)
        mixed_padded = pad_sequence(batch_mixed, batch_first=True)

        # 为了兼容后续的模型输入，再把通道维度加回来，变成 [B, 1, T]
        enroll_padded = enroll_padded.unsqueeze(1)
        mixed_padded = mixed_padded.unsqueeze(1)

        batch_labels = [item["text_label"] for item in batch]
        batch_targets = torch.tensor([item["is_target"] for item in batch], dtype=torch.long)

        return {
            "enroll_wavs": enroll_padded,         # [B, 1, T_enroll]
            "enroll_lengths": enroll_lengths,     # [B]
            "mixed_wavs": mixed_padded,           # [B, 1, T_mixed]
            "mixed_lengths": mixed_lengths,       # [B]
            "text_labels": batch_labels,          # List[str]
            "is_targets": batch_targets,          # [B]
        }

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader


# ====================================================================
# 主程序入口（快速自检）
# ====================================================================

if __name__ == "__main__":
    # 设置日志级别以便在自检时看到详细信息
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    # 使用相对于项目根目录的路径
    project_root = Path(__file__).resolve().parents[2]  # src/data -> src -> project root
    jsonl_file = project_root / "data" / "raw" / "test_set_a" / "labels.jsonl"
    audio_root = project_root / "data" / "raw" / "test_set_a"

    print("=" * 60)
    print("ComplexSpeechDataset 自检")
    print("=" * 60)
    print(f"JSONL 路径: {jsonl_file}")
    print(f"音频根目录: {audio_root}")

    if not jsonl_file.exists():
        print(f"[ERROR] labels.jsonl 未找到: {jsonl_file}")
        print("请先确保数据已迁移：将 datasetA 的内容复制到 data/raw/test_set_a/ 下")
    else:
        # 创建数据集实例
        ds = ComplexSpeechDataset(
            jsonl_path=str(jsonl_file),
            root_dir=str(audio_root),
            sample_rate=16000,
        )

        # 打印统计信息
        stats = ds.get_statistics()
        print(f"\n数据集统计: total={stats['total']}, "
              f"pos={stats['pos']}, neg={stats['neg']}")

        # 加载并展示前 3 条样本
        print("\n前 3 条样本预览:")
        for i in range(min(3, len(ds))):
            sample = ds[i]
            enroll_shape = tuple(sample["enroll_wav"].shape)
            mixed_shape = tuple(sample["mixed_wav"].shape)
            print(f"  样本 #{i}:")
            print(f"    enroll_wav shape: {enroll_shape}")
            print(f"    mixed_wav shape : {mixed_shape}")
            print(f"    text_label      : {sample['text_label'][:60]}{'...' if len(sample['text_label']) > 60 else ''}")
            print(f"    is_target       : {sample['is_target']}")
            print(f"    enroll_text     : {sample['enroll_text']}")

        # 测试 DataLoader
        print("\nDataLoader 测试 (batch_size=2):")
        loader = create_dataloader_from_config(
            jsonl_path=str(jsonl_file),
            root_dir=str(audio_root),
            batch_size=2,
            shuffle=True,
            num_workers=0,  # 主进程调试
        )
        batch = next(iter(loader))
        print(f"  batch keys: {list(batch.keys())}")
        print(f"  batch size: {len(batch['enroll_wavs'])}")
        print(f"  is_targets : {batch['is_targets'].tolist()}")

    print("\n自检完成！")