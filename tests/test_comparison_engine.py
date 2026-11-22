"""Unit tests for the Before-vs-After Comparison & Real-Time Wire Simulation Engine."""

import tempfile
import unittest
from pathlib import Path

from evaluation.comparison_engine import run_full_comparison


class ComparisonEngineTests(unittest.TestCase):
    def test_before_after_board_and_wire_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_full_comparison(
                sample_index=0,
                quantity=100,
                currency="USD",
                output_dir=Path(directory),
            )
            self.assertTrue(result["json"].is_file())
            self.assertTrue(result["png"].is_file())

            report = result["report"]
            m1 = report["module1_board_comparison"]
            self.assertGreater(
                m1["before"]["logistic_failure_probability"],
                m1["after"]["logistic_failure_probability"],
            )
            self.assertEqual(m1["after"]["residual_defect_count"], 0)
            self.assertEqual(m1["after"]["board_status"], "PASS_VERIFIED")

            ov = report["overall_circuit_improvement"]
            self.assertEqual(ov["total_transistor_branches"], 3)
            self.assertLess(
                ov["after_top_ranked_switching_loss_mw"],
                ov["before_total_switching_loss_mw"],
            )
            self.assertGreater(
                ov["after_top_ranked_mean_efficiency_pct"],
                ov["before_mean_branch_efficiency_pct"],
            )
            self.assertEqual(ov["after_total_dc_drive_current_ma"], 0.0)
            self.assertIsNotNone(report["image_to_circuit_synthesis"])

    def test_different_images_synthesize_different_bom_and_netlist_values(self):
        with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
            res_0 = run_full_comparison(sample_index=0, output_dir=Path(dir_a))
            res_7 = run_full_comparison(sample_index=7, output_dir=Path(dir_b))
            params_0 = res_0["report"]["image_to_circuit_synthesis"]["derived_component_parameters"]
            params_7 = res_7["report"]["image_to_circuit_synthesis"]["derived_component_parameters"]
            self.assertNotEqual(params_0, params_7)


if __name__ == "__main__":
    unittest.main()
