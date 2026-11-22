"""Unit tests for Module 3: Rule-based MOSFET replacement recommendations and datasheet evidence."""

import tempfile
import unittest
from pathlib import Path

from recommendations.recommender import (
    DEFAULT_CATALOG_PATH,
    extract_package_from_footprint,
    export_recommendations,
    load_mosfet_catalog,
    recommend_replacements,
)


class RecommendationTests(unittest.TestCase):
    def test_catalog_loads_and_validates(self):
        catalog = load_mosfet_catalog(DEFAULT_CATALOG_PATH)
        self.assertGreaterEqual(len(catalog["candidates"]), 10)
        self.assertEqual(extract_package_from_footprint("footprints:TO-92"), "TO-92")
        self.assertEqual(
            extract_package_from_footprint("Package_TO_SOT_THT:TO-220-3_Horizontal_TabDown"),
            "TO-220",
        )

    def test_q1_n_channel_recommendations_and_pinout_filtering(self):
        report = recommend_replacements(
            reference="Q1",
            min_vds_v=30.0,
            min_id_a=0.1,
            max_vgs_th_v=3.0,
            max_rds_on_ohm=10.0,
            min_pd_w=0.5,
        )
        req = report["operating_requirements"]
        self.assertEqual(req["package"], "TO-92")
        self.assertEqual(req["channel"], "N-channel")
        self.assertEqual(req["pinout"], {"1": "D", "2": "G", "3": "S"})

        eligible_parts = [item["part_number"] for item in report["eligible_candidates"]]
        # ZVN4206A, BS170, VN10LP have complete specs; ZVN3306A is eligible in non-strict mode but flagged for missing pd_max_w
        self.assertIn("ZVN4206A", eligible_parts)
        self.assertIn("BS170", eligible_parts)
        self.assertIn("VN10LP", eligible_parts)
        self.assertIn("ZVN3306A", eligible_parts)

        # 2N7000 has reversed S-G-D pinout, BSS138 is SOT-23, IRLZ44N is TO-220, ZVP3306A is P-channel -> all must be rejected
        self.assertNotIn("2N7000", eligible_parts)
        self.assertNotIn("BSS138", eligible_parts)
        self.assertNotIn("IRLZ44N", eligible_parts)
        self.assertNotIn("ZVP3306A", eligible_parts)

        # Check ZVN3306A has missing spec flag and datasheet evidence attached
        zvn3306 = next(c for c in report["eligible_candidates"] if c["part_number"] == "ZVN3306A")
        self.assertTrue(zvn3306["requires_human_review"])
        self.assertIn("pd_max_w", zvn3306["missing_required_specs"])
        self.assertTrue(zvn3306["datasheet"]["url"].startswith("https://"))
        self.assertTrue(len(zvn3306["datasheet"]["evidence"]) > 20)

    def test_strict_missing_specs_rejects_incomplete_candidates(self):
        report = recommend_replacements(
            reference="Q1",
            min_vds_v=30.0,
            min_id_a=0.1,
            min_pd_w=0.5,
            strict_missing_specs=True,
        )
        eligible_parts = [item["part_number"] for item in report["eligible_candidates"]]
        self.assertNotIn("ZVN3306A", eligible_parts)

    def test_q2_p_channel_recommendations_and_export(self):
        report = recommend_replacements(
            reference="Q2",
            min_vds_v=30.0,
            min_id_a=0.15,
            max_vgs_th_v=3.5,
            max_rds_on_ohm=15.0,
            min_pd_w=0.5,
        )
        self.assertEqual(report["operating_requirements"]["channel"], "P-channel")
        eligible_parts = [item["part_number"] for item in report["eligible_candidates"]]
        self.assertIn("ZVP2106A", eligible_parts)
        self.assertIn("ZVP3306A", eligible_parts)
        self.assertIn("VP2106N3-G", eligible_parts)
        # TP0606N3-G has S-G-D pinout and must be rejected
        self.assertNotIn("TP0606N3-G", eligible_parts)

        with tempfile.TemporaryDirectory() as directory:
            artifacts = export_recommendations(report, Path(directory))
            self.assertTrue(artifacts["json"].is_file())
            self.assertTrue(artifacts["csv"].is_file())


if __name__ == "__main__":
    unittest.main()
