"""Unit tests for Module 4: Cost and performance ranking under matched quantity and currency."""

import tempfile
import unittest
from pathlib import Path

from ranking.cost_performance import (
    DEFAULT_QUOTES_PATH,
    convert_currency,
    export_ranking,
    load_supplier_quotes,
    rank_eligible_candidates,
    unit_price_at_quantity,
)


class CostPerformanceRankingTests(unittest.TestCase):
    def test_quotes_load_and_price_break_normalization(self):
        payload = load_supplier_quotes(DEFAULT_QUOTES_PATH)
        self.assertGreaterEqual(len(payload["quotes"]), 10)
        breaks = [
            {"min_qty": 1, "unit_price": 0.68},
            {"min_qty": 100, "unit_price": 0.44},
            {"min_qty": 1000, "unit_price": 0.31},
        ]
        self.assertEqual(unit_price_at_quantity(breaks, 1), 0.68)
        self.assertEqual(unit_price_at_quantity(breaks, 250), 0.44)
        self.assertEqual(unit_price_at_quantity(breaks, 1500), 0.31)

        rates = payload["exchange_rates_to_usd"]
        self.assertAlmostEqual(convert_currency(1.0, "EUR", "USD", rates), 1.08)
        self.assertAlmostEqual(convert_currency(1.08, "USD", "EUR", rates), 1.0)

    def test_q1_ranking_tradeoffs_and_stale_quote_alert(self):
        report = rank_eligible_candidates(reference="Q1", quantity=100, currency="USD")
        self.assertEqual(report["purchasing_conditions"]["quantity"], 100)
        self.assertEqual(report["purchasing_conditions"]["currency"], "USD")
        self.assertEqual(len(report["ranked_candidates"]), 4)

        ranked_parts = [item["part_number"] for item in report["ranked_candidates"]]
        # BS170 and ZVN4206A should lead the ranking (BS170 is lowest price & highest stock; ZVN4206A has lowest R_DS(on))
        self.assertIn(ranked_parts[0], {"BS170", "ZVN4206A"})
        self.assertIn(ranked_parts[1], {"BS170", "ZVN4206A"})

        # ZVN3306A has a quote from 2026-09-10 (>30 days old) and missing pd_max_w -> both alerts must fire
        self.assertIn("ZVN3306A", report["alerts"]["stale_quote_parts"])
        self.assertIn("ZVN3306A", report["alerts"]["missing_spec_parts"])
        zvn3306 = next(c for c in report["ranked_candidates"] if c["part_number"] == "ZVN3306A")
        self.assertIn("STALE QUOTE ALERT", zvn3306["tradeoff_explanation"])
        self.assertIn("DATASHEET REVIEW REQUIRED", zvn3306["tradeoff_explanation"])

    def test_weight_shifting_and_currency_matching(self):
        # Cost-heavy weighting favors BS170; Performance-heavy weighting favors ZVN4206A (1.5 Ohm vs 5.0 Ohm, 0.6A vs 0.5A)
        cost_First = rank_eligible_candidates(
            reference="Q1", quantity=100, currency="USD",
            weight_cost=0.85, weight_performance=0.10, weight_stock=0.05,
        )
        perf_First = rank_eligible_candidates(
            reference="Q1", quantity=100, currency="USD",
            weight_cost=0.10, weight_performance=0.85, weight_stock=0.05,
        )
        self.assertEqual(cost_First["ranked_candidates"][0]["part_number"], "BS170")
        self.assertEqual(perf_First["ranked_candidates"][0]["part_number"], "ZVN4206A")

        # Verify EUR currency normalization and artifact export
        eur_report = rank_eligible_candidates(reference="Q2", quantity=1000, currency="EUR")
        self.assertEqual(eur_report["purchasing_conditions"]["currency"], "EUR")
        self.assertEqual(len(eur_report["ranked_candidates"]), 3)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = export_ranking(eur_report, Path(directory))
            self.assertTrue(artifacts["json"].is_file())
            self.assertTrue(artifacts["csv"].is_file())


if __name__ == "__main__":
    unittest.main()
