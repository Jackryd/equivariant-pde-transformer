"""Regression checks for evaluating the configurations used in the project."""

import tempfile
import unittest
from pathlib import Path

import torch

from compare_two_checkpoints_test_eval import compute_rollout_summary
from checkpoint_utils import (
    common_evaluation_data_args,
    evaluation_data_args,
    instantiate_from_checkpoint,
)
from navierstokes_official import (
    OfficialTinyEQPDEModel,
    OfficialTinyPDEModel,
    OfficialTinyPaperEQPDEModel,
    _apply_d4_single,
)


class CheckpointWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_checkpoint_roundtrip_for_supported_configurations(self):
        cases = [
            (OfficialTinyPDEModel, {"in_channels": 2}, {"model_variant": "tiny", "history": 2}),
            (
                OfficialTinyEQPDEModel,
                {"eq_token_linear_shared": True, "eq_attention_shared_qkv": True},
                {"model_variant": "tiny_eq", "eq_token_linear_shared": True, "eq_attention_shared_qkv": True},
            ),
            (
                OfficialTinyPaperEQPDEModel,
                {"eq_resample": False},
                {"model_variant": "tiny_eq_paper", "eq_no_resample": True},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for constructor, options, saved_args in cases:
                with self.subTest(args=saved_args):
                    original = constructor(**options).eval()
                    # A zero output head would make a prediction comparison vacuous.
                    with torch.no_grad():
                        for p in original.parameters():
                            if p.requires_grad:
                                p.normal_(0.0, 0.02)
                    path = Path(tmp) / "model.pt"
                    torch.save({"args": saved_args, "model_state_dict": original.state_dict()}, path)
                    restored, _ = instantiate_from_checkpoint(path)
                    channels = saved_args.get("history", 1)
                    x = torch.randn(1, channels, 128, 128)
                    dt = torch.tensor([0.01])
                    with torch.no_grad():
                        torch.testing.assert_close(restored(x, dt), original(x, dt), rtol=0, atol=0)

    def test_evaluation_inherits_saved_split_and_field(self):
        payload = {"args": {"split_seed": 7, "field_idx": 2, "history": 3}}
        self.assertEqual(
            evaluation_data_args(payload),
            {"split_seed": 7, "field_idx": 2, "history": 3, "horizon": 1},
        )
        overridden = evaluation_data_args(payload, split_seed=0, field_idx=0)
        self.assertEqual(overridden["split_seed"], 0)
        self.assertEqual(overridden["field_idx"], 0)

    def test_comparison_rejects_different_training_splits(self):
        payloads = [{"args": {"split_seed": 0}}, {"args": {"split_seed": 7}}]
        with self.assertRaisesRegex(ValueError, "Checkpoint data settings differ"):
            common_evaluation_data_args(payloads)

    def test_rollout_evaluation_preserves_the_history_window(self):
        class LinearTrajectory:
            history = 2
            dt = 1.0
            trajectory_indices = [0]

            def get_trajectory(self, _index):
                return torch.arange(6.0)[:, None, None].expand(6, 4, 4)

        class Extrapolator(torch.nn.Module):
            def forward(self, x, dt):
                return 2 * x[:, -1:] - x[:, -2:-1]

        result = compute_rollout_summary(
            Extrapolator(), LinearTrajectory(), torch.device("cpu"), rollout_steps=4, batch_size=1
        )
        self.assertEqual(result["mean_rmse_t1_to_tN"], 0.0)
        self.assertEqual(result["final_step_rmse"], 0.0)

    def test_paper_model_commutes_with_every_d4_action(self):
        torch.manual_seed(0)
        model = OfficialTinyPaperEQPDEModel().eval()
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.normal_(0.0, 0.02)
            x = torch.randn(1, 1, 128, 128)
            dt = torch.tensor([0.01])
            y = model(x, dt)
            for k in range(4):
                for reflection in (False, True):
                    with self.subTest(rotation=k, reflection=reflection):
                        actual = model(_apply_d4_single(x, k, reflection), dt)
                        expected = _apply_d4_single(y, k, reflection)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_paper_model_rejects_unsupported_relative_bias_ablation(self):
        with self.assertRaisesRegex(ValueError, "canonical pair-orbit"):
            OfficialTinyPaperEQPDEModel(eq_relbias_symmetrize=False)


if __name__ == "__main__":
    unittest.main()
