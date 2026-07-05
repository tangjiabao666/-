#!/usr/bin/env python3
"""TSE-ASR 全部模块完整性验证脚本。依次测试五个核心模块。"""
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

errors = []

def step(n, name):
    print(f"\n{'='*50}")
    print(f"  {n}. {name}")
    print(f"{'='*50}")

# ===== 1. 数据加载 =====
step(1, "数据加载模块")
try:
    from src.data.dataset import ComplexSpeechDataset, create_dataloader_from_config
    ds = ComplexSpeechDataset('data/raw/test_set_a/labels.jsonl', 'data/raw/test_set_a')
    s = ds.get_statistics()
    print(f"  数据集: total={s['total']}, pos={s['pos']}, neg={s['neg']}")
    samp = ds[0]
    print(f"  样本: enroll_wav={tuple(samp['enroll_wav'].shape)}, mixed_wav={tuple(samp['mixed_wav'].shape)}, is_target={samp['is_target']}")
    loader = create_dataloader_from_config('data/raw/test_set_a/labels.jsonl', 'data/raw/test_set_a', batch_size=2, num_workers=0)
    batch = next(iter(loader))
    print(f"  DataLoader keys: {list(batch.keys())}")
    print(f"  enroll_wavs={batch['enroll_wavs'].shape}, mixed_wavs={batch['mixed_wavs'].shape}")
    print(f"  enroll_lengths={batch['enroll_lengths']}, mixed_lengths={batch['mixed_lengths']}")
    print("  [PASS] 数据加载OK")
except Exception as e:
    print(f"  [FAIL] {e}")
    errors.append(f"数据加载: {e}")

# ===== 2. 声纹提取器 =====
step(2, "声纹提取器 (LightSpeakerEncoder)")
try:
    from src.models.speaker_encoder.encoder import LightSpeakerEncoder
    spk = LightSpeakerEncoder(embedding_dim=256, n_mels=80, channels=256)
    p = sum(p.numel() for p in spk.parameters())
    print(f"  参数量: {p:,}")
    spk_emb = spk(batch['enroll_wavs'], batch['enroll_lengths'])
    print(f"  输入: {batch['enroll_wavs'].shape} -> 输出: {spk_emb.shape}")
    print(f"  L2范数: {spk_emb.norm(p=2, dim=1).tolist()}")
    print("  [PASS] 声纹提取器OK")
except Exception as e:
    print(f"  [FAIL] {e}")
    errors.append(f"声纹提取器: {e}")

# ===== 3. TSE 提取器 =====
step(3, "TSE 提取器 (LightTSExtractor)")
try:
    from src.models.tse_network.extractor import LightTSExtractor
    tse = LightTSExtractor(feature_dim=256, hidden_dim=512, emb_dim=256, repeats=2)
    p2 = sum(p.numel() for p in tse.parameters())
    print(f"  参数量: {p2:,}")
    clean = tse(batch['mixed_wavs'], spk_emb)
    print(f"  输入: mixed={batch['mixed_wavs'].shape}, spk={spk_emb.shape} -> 输出: {clean.shape}")
    assert clean.shape == batch['mixed_wavs'].shape, "波形形状不匹配!"
    print("  [PASS] TSE 提取器OK")
except Exception as e:
    print(f"  [FAIL] {e}")
    errors.append(f"TSE 提取器: {e}")

# ===== 4. 联合模型 =====
step(4, "联合模型 (JointTSEASR)")
try:
    from src.models.joint_model import JointTSEASR
    model = JointTSEASR(vocab_size=688, spk_emb_dim=256, spk_channels=256,
                        tse_feature_dim=256, tse_hidden_dim=512, tse_repeats=2,
                        asr_conformer_dim=256, asr_conformer_layers=4)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_p:,}")
    outputs = model(enroll_wavs=batch['enroll_wavs'], enroll_lengths=batch['enroll_lengths'],
                    mixed_wavs=batch['mixed_wavs'], mixed_lengths=batch['mixed_lengths'])
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={list(v.shape)}")
        else:
            print(f"  {k}: {v}")
    print("  [PASS] 联合模型OK")
except Exception as e:
    print(f"  [FAIL] {e}")
    errors.append(f"联合模型: {e}")

# ===== 5. 损失函数 =====
step(5, "损失函数 (JointLoss)")
try:
    from src.utils.loss import JointLoss
    loss_fn = JointLoss()
    tokens = torch.tensor([1,2,3,4,5, 6,7,8], dtype=torch.long)
    tgt_lens = torch.tensor([5, 3], dtype=torch.long)
    is_tgt = torch.tensor([1, 1], dtype=torch.long)
    losses = loss_fn(outputs['clean_wavs'], batch['mixed_wavs'],
                     outputs['asr_logits'], tokens,
                     outputs['asr_lengths'], tgt_lens,
                     outputs['reject_logits'], is_tgt,
                     batch['mixed_lengths'])
    print(f"  total={losses['total_loss'].item():.4f}, tse={losses['tse_loss'].item():.4f}, "
          f"asr={losses['asr_loss'].item():.4f}, reject={losses['reject_loss'].item():.4f}")
    print("  [PASS] 损失函数OK")
except Exception as e:
    print(f"  [FAIL] {e}")
    errors.append(f"损失函数: {e}")

# ===== 总结 =====
print(f"\n{'='*50}")
if errors:
    print(f"  FAILED: {len(errors)} 个模块验证失败")
    for e in errors:
        print(f"    - {e}")
else:
    print(f"  全部 5 个模块验证通过! 模型可以上云服务器训练。")
print(f"{'='*50}")