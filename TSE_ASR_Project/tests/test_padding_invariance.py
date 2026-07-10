import unittest

import torch

from src.models.joint_model import RejectionHead, WaveformFusionGate


class PaddingInvarianceTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.short_len = 16000
        self.long_len = 26000
        self.short_mixed = torch.randn(1, 1, self.short_len) * 0.03
        self.short_tse = torch.randn(1, 1, self.short_len) * 0.005
        self.short_emb = torch.randn(1, 256)

        self.batch_mixed = torch.zeros(2, 1, self.long_len)
        self.batch_tse = torch.zeros(2, 1, self.long_len)
        self.batch_mixed[0, :, : self.short_len] = self.short_mixed
        self.batch_tse[0, :, : self.short_len] = self.short_tse
        self.batch_mixed[1] = torch.randn(1, self.long_len) * 0.03
        self.batch_tse[1] = torch.randn(1, self.long_len) * 0.005
        self.batch_emb = torch.cat([self.short_emb, torch.randn(1, 256)], dim=0)
        self.batch_lengths = torch.tensor([self.short_len, self.long_len])

    def test_fusion_gate_ignores_batch_padding(self) -> None:
        gate = WaveformFusionGate().eval()
        _, single_value = gate(
            self.short_mixed,
            self.short_tse,
            self.short_emb,
            torch.tensor([self.short_len]),
        )
        _, batch_value = gate(
            self.batch_mixed,
            self.batch_tse,
            self.batch_emb,
            self.batch_lengths,
        )
        torch.testing.assert_close(single_value[0], batch_value[0], rtol=1e-6, atol=1e-7)

    def test_rejection_head_ignores_batch_padding(self) -> None:
        head = RejectionHead().eval()
        short_clean = 0.15 * self.short_tse + 0.85 * self.short_mixed
        batch_clean = 0.15 * self.batch_tse + 0.85 * self.batch_mixed
        single_logits = head(
            short_clean,
            self.short_emb,
            mixed_wavs=self.short_mixed,
            extracted_wavs=self.short_tse,
            lengths=torch.tensor([self.short_len]),
        )
        batch_logits = head(
            batch_clean,
            self.batch_emb,
            mixed_wavs=self.batch_mixed,
            extracted_wavs=self.batch_tse,
            lengths=self.batch_lengths,
        )
        torch.testing.assert_close(single_logits[0], batch_logits[0], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
