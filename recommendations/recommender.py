"""Module 3: Rule-based MOSFET replacement recommendations linked to Module 2 circuit mapping.

Filters catalog candidates by package/footprint, pinout mapping (Drain/Gate/Source), channel
polarity, and electrical operating limits (V_DS, I_D, V_GS(th), R_DS(on), P_D). Flags missing
specifications and provides datasheet evidence for human review.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from components.circuit_map import DEFAULT_BOM_PATH, DEFAULT_NETLIST_PATH, build_circuit_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data/inputs/mosfet_catalog.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/recommendations"

KNOWN_N_CHANNEL_PARTS = {
    "BC237", "BC547", "BC548", "2N2222", "2N3904", "2N7000", "BS170", "VN10LP", "ZVN4206A", "ZVN3306A", "BSS138", "IRLZ44N",
}
KNOWN_P_CHANNEL_PARTS = {
    "BC307", "BC557", "BC558", "2N2907", "2N3906", "ZVP3306A", "ZVP2106A", "TP0606N3-G", "VP2106N3-G",
}
ALL_SPEC_KEYS = (
    "vds_max_v",
    "id_max_a",
    "vgs_max_v",
    "vgs_th_max_v",
    "rds_on_max_ohm",
    "pd_max_w",
    "qg_typ_nc",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mosfet_catalog(catalog_path=DEFAULT_CATALOG_PATH):
    """Load and validate the curated MOSFET catalog snapshot."""
    catalog_path = Path(catalog_path)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing MOSFET catalog file: {catalog_path}")

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Catalog {catalog_path} must contain a non-empty 'candidates' list")

    seen = set()
    for idx, item in enumerate(candidates, 1):
        part = (item.get("part_number") or "").strip()
        if not part:
            raise ValueError(f"Catalog entry #{idx} is missing 'part_number'")
        if part in seen:
            raise ValueError(f"Duplicate part_number in MOSFET catalog: {part}")
        seen.add(part)
        if item.get("channel") not in {"N-channel", "P-channel"}:
            raise ValueError(f"Catalog part {part}: channel must be 'N-channel' or 'P-channel'")
        if not item.get("package"):
            raise ValueError(f"Catalog part {part}: missing 'package'")
        pinout = item.get("pinout")
        if not isinstance(pinout, dict) or set(pinout.values()) != {"D", "G", "S"}:
            raise ValueError(f"Catalog part {part}: pinout must map pins to 'D', 'G', and 'S'")
        datasheet = item.get("datasheet") or {}
        if not datasheet.get("url") or not datasheet.get("evidence"):
            raise ValueError(f"Catalog part {part}: missing datasheet URL or evidence")

    return payload


def extract_package_from_footprint(footprint):
    """Normalize a KiCad footprint string (e.g., 'footprints:TO-92') to a standard package code."""
    text = (footprint or "").upper()
    for pkg in ("TO-92", "TO-220", "SOT-23", "SOIC-8", "DPAK", "D2PAK"):
        if pkg in text:
            return pkg
    if ":" in text:
        return text.split(":", 1)[1].split("_", 1)[0]
    return text or "UNKNOWN"


def infer_pinout_from_component(component):
    """Map schematic pin functions (D/G/S or BJT C/B/E) to required MOSFET terminal roles (D, G, S)."""
    role_by_token = {
        "D": "D", "DRAIN": "D", "C": "D", "COLLECTOR": "D",
        "G": "G", "GATE": "G", "B": "G", "BASE": "G",
        "S": "S", "SOURCE": "S", "E": "S", "EMITTER": "S",
    }
    pinout = {}
    for pin_entry in component.get("pins", []):
        pin_num = str(pin_entry.get("pin", "")).strip()
        raw_fn = str(pin_entry.get("pinfunction", "")).strip().upper()
        token = raw_fn.split("_", 1)[0].split("/", 1)[0]
        if token in role_by_token:
            pinout[pin_num] = role_by_token[token]

    if set(pinout.values()) == {"D", "G", "S"} and len(pinout) == 3:
        return pinout
    return {"1": "D", "2": "G", "3": "S"}


def infer_channel_from_component(component):
    """Infer N-channel vs P-channel polarity from known transistor part numbers or net topology."""
    part = (component.get("part_number") or component.get("value") or "").strip().upper()
    if part in KNOWN_N_CHANNEL_PARTS:
        return "N-channel"
    if part in KNOWN_P_CHANNEL_PARTS:
        return "P-channel"
    for pin_entry in component.get("pins", []):
        fn = str(pin_entry.get("pinfunction", "")).upper()
        net = str(pin_entry.get("net_name", "")).upper()
        if fn.startswith(("E", "S")) and net in {"GND", "VSS", "0V"}:
            return "N-channel"
        if fn.startswith(("E", "S")) and any(k in net for k in ("VCC", "VDD", "VPP", "+5V", "+12V")):
            return "P-channel"
    return "N-channel"


def evaluate_candidate(candidate, requirements):
    """Evaluate a single catalog MOSFET against package, pinout, polarity, and electrical requirements."""
    specs = candidate.get("specs") or {}
    checks = {}
    violations = []
    missing_required_specs = []
    missing_optional_specs = []

    for key in ALL_SPEC_KEYS:
        if specs.get(key) is None:
            missing_optional_specs.append(key)

    # 1. Package compatibility
    req_pkg = requirements["package"].upper()
    cand_pkg = candidate["package"].upper()
    pkg_ok = cand_pkg == req_pkg
    checks["package"] = {"required": req_pkg, "candidate": cand_pkg, "passed": pkg_ok}
    if not pkg_ok:
        violations.append(f"Package mismatch: requires {req_pkg}, candidate is {cand_pkg}")

    # 2. Channel polarity compatibility
    req_ch = requirements["channel"]
    cand_ch = candidate["channel"]
    ch_ok = cand_ch == req_ch
    checks["channel"] = {"required": req_ch, "candidate": cand_ch, "passed": ch_ok}
    if not ch_ok:
        violations.append(f"Channel mismatch: requires {req_ch}, candidate is {cand_ch}")

    # 3. Pinout compatibility
    req_pinout = {str(k): v for k, v in requirements["pinout"].items()}
    cand_pinout = {str(k): v for k, v in candidate["pinout"].items()}
    pin_ok = cand_pinout == req_pinout
    checks["pinout"] = {"required": req_pinout, "candidate": cand_pinout, "passed": pin_ok}
    if not pin_ok:
        violations.append(f"Pinout mismatch: requires {req_pinout}, candidate has {cand_pinout}")

    # 4. Electrical limits: V_DS, I_D, V_GS(th), R_DS(on), P_D
    limit_rules = [
        ("vds_max_v", "min_vds_v", ">=", "V_DS max (V)"),
        ("id_max_a", "min_id_a", ">=", "I_D max (A)"),
        ("vgs_th_max_v", "max_vgs_th_v", "<=", "V_GS(th) max (V)"),
        ("rds_on_max_ohm", "max_rds_on_ohm", "<=", "R_DS(on) max (Ohm)"),
        ("pd_max_w", "min_pd_w", ">=", "P_D max (W)"),
    ]

    for spec_key, req_key, operator, label in limit_rules:
        req_val = requirements.get(req_key)
        if req_val is None:
            continue
        cand_val = specs.get(spec_key)
        if cand_val is None:
            missing_required_specs.append(spec_key)
            checks[spec_key] = {
                "required": f"{operator} {req_val}",
                "candidate": None,
                "passed": False,
                "status": "missing_spec",
            }
            if requirements.get("strict_missing_specs", False):
                violations.append(f"Missing required specification: {label} ({spec_key})")
        else:
            passed = (cand_val >= req_val) if operator == ">=" else (cand_val <= req_val)
            checks[spec_key] = {
                "required": f"{operator} {req_val}",
                "candidate": cand_val,
                "passed": passed,
                "status": "ok" if passed else "limit_violation",
            }
            if not passed:
                violations.append(
                    f"{label} violation: requires {operator} {req_val}, candidate is {cand_val}"
                )

    eligible = len(violations) == 0
    requires_human_review = len(missing_required_specs) > 0 or len(missing_optional_specs) > 0

    # Compute electrical suitability score for tie-breaking / preview before Module 4 cost ranking
    vds_ratio = (specs.get("vds_max_v") or 0.0) / max(requirements.get("min_vds_v") or 1.0, 1.0)
    id_ratio = (specs.get("id_max_a") or 0.0) / max(requirements.get("min_id_a") or 0.05, 0.05)
    rds = specs.get("rds_on_max_ohm")
    rds_score = (1.0 / max(rds, 0.01)) if rds is not None else 0.0
    suitability_score = round(min(vds_ratio, 5.0) * 10.0 + min(id_ratio, 10.0) * 10.0 + min(rds_score, 10.0) * 5.0, 2)

    return {
        "part_number": candidate["part_number"],
        "manufacturer": candidate["manufacturer"],
        "channel": candidate["channel"],
        "package": candidate["package"],
        "pinout": candidate["pinout"],
        "specs": specs,
        "eligible": eligible,
        "requires_human_review": requires_human_review,
        "missing_required_specs": missing_required_specs,
        "missing_optional_specs": missing_optional_specs,
        "violations": violations,
        "checks": checks,
        "electrical_suitability_score": suitability_score if eligible else 0.0,
        "datasheet": candidate["datasheet"],
    }


def recommend_replacements(
    reference="Q1",
    circuit_map=None,
    catalog_path=DEFAULT_CATALOG_PATH,
    bom_path=DEFAULT_BOM_PATH,
    netlist_path=DEFAULT_NETLIST_PATH,
    min_vds_v=25.0,
    min_id_a=0.1,
    max_vgs_th_v=3.5,
    max_rds_on_ohm=15.0,
    min_pd_w=0.3,
    channel_override=None,
    package_override=None,
    strict_missing_specs=False,
):
    """Filter catalog MOSFETs for a component in the Module 2 circuit map."""
    if circuit_map is None:
        circuit_map = build_circuit_map(bom_path, netlist_path)
    catalog = load_mosfet_catalog(catalog_path)

    target_ref = str(reference).strip().upper()
    component = next(
        (c for c in circuit_map["components"] if c["reference"].upper() == target_ref),
        None,
    )
    if component is None:
        raise ValueError(f"Component reference '{reference}' not found in Module 2 circuit map")

    requirements = {
        "reference": component["reference"],
        "original_value": component["value"],
        "original_footprint": component["footprint"],
        "package": package_override or extract_package_from_footprint(component["footprint"]),
        "channel": channel_override or infer_channel_from_component(component),
        "pinout": infer_pinout_from_component(component),
        "min_vds_v": min_vds_v,
        "min_id_a": min_id_a,
        "max_vgs_th_v": max_vgs_th_v,
        "max_rds_on_ohm": max_rds_on_ohm,
        "min_pd_w": min_pd_w,
        "strict_missing_specs": strict_missing_specs,
    }

    evaluations = [
        evaluate_candidate(candidate, requirements)
        for candidate in catalog["candidates"]
    ]
    eligible = sorted(
        [item for item in evaluations if item["eligible"]],
        key=lambda item: (
            len(item["missing_required_specs"]),
            len(item["missing_optional_specs"]),
            -item["electrical_suitability_score"],
        ),
    )
    rejected = [item for item in evaluations if not item["eligible"]]

    return {
        "catalog_version": catalog.get("catalog_version", "unknown"),
        "catalog_path": str(Path(catalog_path).resolve()),
        "catalog_sha256": sha256(catalog_path),
        "target_component": {
            "reference": component["reference"],
            "category": component["category"],
            "value": component["value"],
            "part_number": component["part_number"],
            "footprint": component["footprint"],
            "pins": component["pins"],
        },
        "operating_requirements": requirements,
        "summary": {
            "total_catalog_candidates": len(evaluations),
            "eligible_count": len(eligible),
            "rejected_count": len(rejected),
            "flagged_missing_specs_count": sum(
                1 for item in eligible if item["requires_human_review"]
            ),
        },
        "eligible_candidates": eligible,
        "rejected_candidates": rejected,
        "disclaimer": (
            "Explicit rule-based filtering only. Flagged missing specifications and datasheet "
            "citations require human engineering review before board assembly."
        ),
    }


def export_recommendations(report, output_dir=DEFAULT_OUTPUT_DIR):
    """Export Module 3 recommendation report and CSV tables for downstream Module 4 ranking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = report["target_component"]["reference"].lower()

    json_path = output_dir / f"recommendations_{ref}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / f"candidates_{ref}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "part_number",
                "manufacturer",
                "channel",
                "package",
                "pinout",
                "eligible",
                "requires_human_review",
                "missing_required_specs",
                "missing_optional_specs",
                "vds_max_v",
                "id_max_a",
                "vgs_th_max_v",
                "rds_on_max_ohm",
                "pd_max_w",
                "electrical_suitability_score",
                "violations",
                "datasheet_url",
                "datasheet_revision",
            ],
        )
        writer.writeheader()
        for item in report["eligible_candidates"] + report["rejected_candidates"]:
            specs = item["specs"]
            pin_str = "-".join(f"{k}:{v}" for k, v in sorted(item["pinout"].items()))
            writer.writerow({
                "part_number": item["part_number"],
                "manufacturer": item["manufacturer"],
                "channel": item["channel"],
                "package": item["package"],
                "pinout": pin_str,
                "eligible": item["eligible"],
                "requires_human_review": item["requires_human_review"],
                "missing_required_specs": ";".join(item["missing_required_specs"]),
                "missing_optional_specs": ";".join(item["missing_optional_specs"]),
                "vds_max_v": specs.get("vds_max_v"),
                "id_max_a": specs.get("id_max_a"),
                "vgs_th_max_v": specs.get("vgs_th_max_v"),
                "rds_on_max_ohm": specs.get("rds_on_max_ohm"),
                "pd_max_w": specs.get("pd_max_w"),
                "electrical_suitability_score": item["electrical_suitability_score"],
                "violations": " | ".join(item["violations"]),
                "datasheet_url": item["datasheet"]["url"],
                "datasheet_revision": item["datasheet"]["revision"],
            })

    return {"json": json_path, "csv": csv_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="Q1", help="Target component reference from Module 2 (e.g., Q1, Q2, Q3)")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--bom", type=Path, default=DEFAULT_BOM_PATH)
    parser.add_argument("--netlist", type=Path, default=DEFAULT_NETLIST_PATH)
    parser.add_argument("--min-vds", type=float, default=25.0, help="Minimum V_DS breakdown voltage (V)")
    parser.add_argument("--min-id", type=float, default=0.1, help="Minimum continuous drain current I_D (A)")
    parser.add_argument("--max-vgs-th", type=float, default=3.5, help="Maximum gate threshold voltage V_GS(th) (V)")
    parser.add_argument("--max-rds-on", type=float, default=15.0, help="Maximum on-resistance R_DS(on) (Ohm)")
    parser.add_argument("--min-pd", type=float, default=0.3, help="Minimum power dissipation P_D (W)")
    parser.add_argument("--strict-missing-specs", action="store_true",
                        help="Reject candidates with any missing required specification instead of flagging for review")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = recommend_replacements(
        reference=args.ref,
        catalog_path=args.catalog,
        bom_path=args.bom,
        netlist_path=args.netlist,
        min_vds_v=args.min_vds,
        min_id_a=args.min_id,
        max_vgs_th_v=args.max_vgs_th,
        max_rds_on_ohm=args.max_rds_on,
        min_pd_w=args.min_pd,
        strict_missing_specs=args.strict_missing_specs,
    )
    artifacts = export_recommendations(report, args.output_dir)

    req = report["operating_requirements"]
    summary = report["summary"]
    print(
        f"Target: {req['reference']} (Original: {req['original_value']}, Footprint: {req['original_footprint']}) -> "
        f"Required Package: {req['package']} | Channel: {req['channel']} | Pinout: {req['pinout']}"
    )
    print(
        f"Catalog Candidates: {summary['total_catalog_candidates']} | "
        f"Eligible: {summary['eligible_count']} | "
        f"Rejected: {summary['rejected_count']} | "
        f"Flagged for Missing Specs: {summary['flagged_missing_specs_count']}"
    )
    print("\nEligible MOSFET Candidates:")
    for idx, item in enumerate(report["eligible_candidates"], 1):
        specs = item["specs"]
        flags = []
        if item["missing_required_specs"]:
            flags.append(f"MISSING_REQ={item['missing_required_specs']}")
        if item["missing_optional_specs"]:
            flags.append(f"MISSING_OPT={item['missing_optional_specs']}")
        flag_str = f" [FLAGGED: {', '.join(flags)}]" if flags else " [COMPLETE SPECS]"
        print(
            f"  {idx}. {item['part_number']} ({item['manufacturer']}){flag_str}\n"
            f"     V_DS={specs['vds_max_v']}V | I_D={specs['id_max_a']}A | "
            f"V_GS(th)={specs['vgs_th_max_v']}V | R_DS(on)={specs['rds_on_max_ohm']} Ohm | "
            f"P_D={specs['pd_max_w']}W\n"
            f"     Datasheet ({item['datasheet']['revision']}): {item['datasheet']['url']}\n"
            f"     Evidence: {item['datasheet']['evidence']}"
        )

    print(f"\nExported JSON report: {artifacts['json']}")
    print(f"Exported CSV table:   {artifacts['csv']}")


if __name__ == "__main__":
    main()
