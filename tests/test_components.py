"""Unit tests for Module 2: BOM CSV parser, KiCad S-expression netlist parser, and circuit mapping."""

import tempfile
import unittest
from pathlib import Path

from components.circuit_map import (
    DEFAULT_BOM_PATH,
    DEFAULT_NETLIST_PATH,
    build_circuit_map,
    classify_reference,
    export_circuit_map,
    parse_bom_csv,
    parse_kicad_netlist,
    parse_sexpr,
)


class ComponentCircuitMapTests(unittest.TestCase):
    def test_parse_project_bom_and_netlist(self):
        bom = parse_bom_csv(DEFAULT_BOM_PATH)
        self.assertEqual(len(bom), 56)
        self.assertEqual(bom["U3"]["value"], "7805")
        self.assertEqual(bom["Q1"]["value"], "BC237")
        self.assertEqual(classify_reference("Q1"), "Transistor / MOSFET")

        netlist = parse_kicad_netlist(DEFAULT_NETLIST_PATH)
        self.assertEqual(len(netlist["components"]), 63)
        self.assertEqual(len(netlist["nets"]), 111)

        circuit_map = build_circuit_map(DEFAULT_BOM_PATH, DEFAULT_NETLIST_PATH)
        val = circuit_map["validation"]
        self.assertTrue(val["is_valid"])
        self.assertEqual(val["matched_component_count"], 56)
        self.assertEqual(val["bom_only_references"], [])
        self.assertEqual(val["netlist_only_references"], [])
        self.assertEqual(
            val["netlist_non_bom_hardware"],
            ["JP1", "P101", "P102", "P103", "P104", "P105", "P106"],
        )
        self.assertEqual(val["value_mismatches"], [])
        self.assertEqual(val["footprint_mismatches"], [])
        self.assertEqual(val["active_nets"], 34)
        self.assertEqual(val["unconnected_nets"], 77)

        q1 = next(c for c in circuit_map["components"] if c["reference"] == "Q1")
        self.assertEqual(q1["pin_count"], 3)
        pin_nets = {p["pin"]: p["net_name"] for p in q1["pins"]}
        self.assertEqual(pin_nets, {"1": "Net-(D11-K)", "2": "Net-(Q1-B)", "3": "GND"})

    def test_export_artifacts_and_detect_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bom_path = directory / "bom.csv"
            net_path = directory / "board.net"
            bom_path.write_text(
                '"Reference","Value","Datasheet","Footprint"\n'
                '"R1","10K","","Resistor_THT:R_Axial"\n'
                '"Q1","2N7000","http://example.com/2n7000.pdf","Package_TO_SOT_THT:TO-92"\n',
                encoding="utf-8",
            )
            net_path.write_text(
                '(export (version "E")\n'
                '  (design (source "test.kicad_sch") (date "2026-11-21") (tool "Eeschema"))\n'
                '  (components\n'
                '    (comp (ref "R1") (value "22K") (footprint "Resistor_THT:R_Axial")\n'
                '      (libsource (lib "Device") (part "R")))\n'
                '    (comp (ref "Q1") (value "2N7000") (footprint "Package_TO_SOT_THT:TO-92")\n'
                '      (libsource (lib "Transistor_FET") (part "2N7000")))\n'
                '  )\n'
                '  (nets\n'
                '    (net (code "1") (name "VCC") (class "Default")\n'
                '      (node (ref "R1") (pin "1") (pintype "passive"))\n'
                '      (node (ref "Q1") (pin "1") (pinfunction "D") (pintype "passive")))\n'
                '    (net (code "2") (name "GND") (class "Default")\n'
                '      (node (ref "R1") (pin "2") (pintype "passive"))\n'
                '      (node (ref "Q1") (pin "3") (pinfunction "S") (pintype "passive")))\n'
                '  )\n'
                ')\n',
                encoding="utf-8",
            )
            circuit_map = build_circuit_map(bom_path, net_path)
            self.assertFalse(circuit_map["validation"]["is_valid"])
            self.assertEqual(len(circuit_map["validation"]["value_mismatches"]), 1)
            self.assertEqual(circuit_map["validation"]["value_mismatches"][0]["reference"], "R1")

            out_dir = directory / "out"
            artifacts = export_circuit_map(circuit_map, out_dir)
            self.assertTrue(artifacts["circuit_map_json"].is_file())
            self.assertTrue(artifacts["components_csv"].is_file())
            self.assertTrue(artifacts["pin_connections_csv"].is_file())

    def test_invalid_sexpr_and_duplicate_bom_refs(self):
        with self.assertRaises(ValueError):
            parse_sexpr("(export (unclosed)")
        with tempfile.TemporaryDirectory() as directory:
            bom_path = Path(directory) / "dup.csv"
            bom_path.write_text(
                '"Reference","Value","Datasheet","Footprint"\n'
                '"R1","10K","","FP1"\n'
                '"R1","10K","","FP1"\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                parse_bom_csv(bom_path)


if __name__ == "__main__":
    unittest.main()
