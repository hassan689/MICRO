"""Parse and cross-validate a BOM CSV and KiCad S-expression netlist to build a circuit map."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOM_PATH = PROJECT_ROOT / "data/inputs/bom.csv"
DEFAULT_NETLIST_PATH = PROJECT_ROOT / "data/inputs/board.net"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/components"

CATEGORY_BY_PREFIX = {
    "U": "IC / Microcontroller / Regulator",
    "Q": "Transistor / MOSFET",
    "D": "Diode / LED",
    "R": "Resistor",
    "RV": "Potentiometer / Variable Resistor",
    "C": "Capacitor",
    "L": "Inductor",
    "J": "Connector",
    "JP": "Solder Jumper",
    "P": "Header / Socket / Connector",
}


def is_non_bom_hardware(comp):
    """Identify PCB features (solder pads, mounting holes, fiducials) excluded from BOMs."""
    fp = (comp.get("footprint") or "").lower()
    part = (comp.get("library_part") or "").lower()
    val = (comp.get("value") or "").lower()
    return (
        "mountinghole" in fp
        or "mounting_hole" in part
        or "solderjumper" in fp
        or (part == "jumper" and val == "jumper")
        or "fiducial" in fp
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_reference(ref):
    match = re.match(r"^([A-Za-z]+)", str(ref).strip())
    prefix = match.group(1).upper() if match else ""
    return CATEGORY_BY_PREFIX.get(prefix, "Other")


def reference_sort_key(ref):
    match = re.match(r"^([A-Za-z]+)(\d+)(.*)$", str(ref).strip())
    if match:
        return (match.group(1).upper(), int(match.group(2)), match.group(3))
    return (str(ref).strip(), 0, "")


def pin_sort_key(pin_str):
    text = str(pin_str).strip()
    if text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def parse_bom_csv(bom_path=DEFAULT_BOM_PATH):
    """Parse and validate a BOM CSV file."""
    bom_path = Path(bom_path)
    if not bom_path.is_file():
        raise FileNotFoundError(f"Missing BOM CSV file: {bom_path}")

    with bom_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty BOM CSV or missing header row: {bom_path}")
        required_columns = {"Reference", "Value", "Footprint"}
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"BOM CSV {bom_path} missing required columns: {sorted(missing_columns)}"
            )

        components = {}
        duplicates = []
        for row_number, row in enumerate(reader, 2):
            ref = (row.get("Reference") or "").strip()
            if not ref:
                raise ValueError(f"{bom_path}:{row_number}: empty component Reference")
            if ref in components:
                duplicates.append(ref)
                continue
            value = (row.get("Value") or "").strip()
            footprint = (row.get("Footprint") or "").strip()
            datasheet = (row.get("Datasheet") or "").strip()
            part_number = (
                row.get("PartNumber")
                or row.get("Part Number")
                or row.get("MPN")
                or value
            ).strip()
            components[ref] = {
                "reference": ref,
                "category": classify_reference(ref),
                "value": value,
                "part_number": part_number,
                "footprint": footprint,
                "datasheet": datasheet,
            }

    if duplicates:
        raise ValueError(f"Duplicate component references in BOM {bom_path}: {duplicates}")
    if not components:
        raise ValueError(f"No component rows found in BOM {bom_path}")

    return components


def parse_sexpr(text):
    """Tokenize and parse a KiCad S-expression string into nested Python lists."""
    tokens = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
        elif char == "(":
            tokens.append("(")
            index += 1
        elif char == ")":
            tokens.append(")")
            index += 1
        elif char == '"':
            index += 1
            buffer = []
            while index < length:
                if text[index] == "\\" and index + 1 < length:
                    buffer.append(text[index + 1])
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    buffer.append(text[index])
                    index += 1
            else:
                raise ValueError("Unterminated quoted string in S-expression")
            tokens.append("".join(buffer))
        else:
            start = index
            while index < length and not text[index].isspace() and text[index] not in "()\"":
                index += 1
            tokens.append(text[start:index])

    if not tokens:
        raise ValueError("Empty S-expression input")

    stack = [[]]
    for token in tokens:
        if token == "(":
            node = []
            stack[-1].append(node)
            stack.append(node)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("Unbalanced closing parenthesis in S-expression")
            stack.pop()
        else:
            stack[-1].append(token)

    if len(stack) != 1:
        raise ValueError("Unclosed parenthesis in S-expression")
    if len(stack[0]) != 1 or not isinstance(stack[0][0], list):
        raise ValueError("Root of S-expression must be a single list")
    return stack[0][0]


def _find_child(node, tag):
    for item in node[1:]:
        if isinstance(item, list) and item and item[0] == tag:
            return item
    return None


def _find_children(node, tag):
    return [
        item for item in node[1:]
        if isinstance(item, list) and item and item[0] == tag
    ]


def _scalar_value(node, tag, default=""):
    child = _find_child(node, tag)
    if child and len(child) >= 2 and isinstance(child[1], str):
        return child[1].strip()
    return default


def parse_kicad_netlist(netlist_path=DEFAULT_NETLIST_PATH):
    """Parse a KiCad .net S-expression file into design info, components, and nets."""
    netlist_path = Path(netlist_path)
    if not netlist_path.is_file():
        raise FileNotFoundError(f"Missing KiCad netlist file: {netlist_path}")

    tree = parse_sexpr(netlist_path.read_text(encoding="utf-8"))
    if not tree or tree[0] != "export":
        raise ValueError(f"Invalid KiCad netlist root tag in {netlist_path}")

    design_node = _find_child(tree, "design")
    design_info = {
        "version": _scalar_value(tree, "version"),
        "source": _scalar_value(design_node, "source") if design_node else "",
        "date": _scalar_value(design_node, "date") if design_node else "",
        "tool": _scalar_value(design_node, "tool") if design_node else "",
    }

    components_node = _find_child(tree, "components")
    if components_node is None:
        raise ValueError(f"Netlist {netlist_path} is missing a (components ...) section")

    netlist_components = {}
    for comp in _find_children(components_node, "comp"):
        ref = _scalar_value(comp, "ref")
        if not ref:
            continue
        value = _scalar_value(comp, "value")
        footprint = _scalar_value(comp, "footprint")
        datasheet = _scalar_value(comp, "datasheet")
        libsource = _find_child(comp, "libsource")
        lib_name = _scalar_value(libsource, "lib") if libsource else ""
        part_name = _scalar_value(libsource, "part") if libsource else ""

        fields = {}
        fields_node = _find_child(comp, "fields")
        if fields_node:
            for field_item in _find_children(fields_node, "field"):
                field_name = _scalar_value(field_item, "name")
                field_val = ""
                if len(field_item) >= 3 and isinstance(field_item[-1], str):
                    field_val = field_item[-1].strip()
                elif _scalar_value(field_item, "value"):
                    field_val = _scalar_value(field_item, "value")
                if field_name:
                    fields[field_name] = field_val

        properties = {}
        for prop_item in _find_children(comp, "property"):
            prop_name = _scalar_value(prop_item, "name")
            prop_val = _scalar_value(prop_item, "value")
            if prop_name:
                properties[prop_name] = prop_val

        declared_pins = []
        units_node = _find_child(comp, "units")
        if units_node:
            for unit_item in _find_children(units_node, "unit"):
                pins_node = _find_child(unit_item, "pins")
                if pins_node:
                    for pin_item in _find_children(pins_node, "pin"):
                        pin_num = _scalar_value(pin_item, "num")
                        if pin_num and pin_num not in declared_pins:
                            declared_pins.append(pin_num)

        part_number = (
            fields.get("PartNumber")
            or fields.get("Part Number")
            or fields.get("MPN")
            or value
            or part_name
        )
        netlist_components[ref] = {
            "reference": ref,
            "category": classify_reference(ref),
            "value": value,
            "part_number": part_number,
            "footprint": footprint or fields.get("Footprint", ""),
            "datasheet": datasheet or fields.get("Datasheet", ""),
            "library": lib_name,
            "library_part": part_name,
            "sheet_name": properties.get("Sheetname", ""),
            "sheet_file": properties.get("Sheetfile", ""),
            "declared_pins": sorted(declared_pins, key=pin_sort_key),
        }

    nets_node = _find_child(tree, "nets")
    if nets_node is None:
        raise ValueError(f"Netlist {netlist_path} is missing a (nets ...) section")

    nets = []
    for net_item in _find_children(nets_node, "net"):
        code = _scalar_value(net_item, "code")
        name = _scalar_value(net_item, "name")
        net_class = _scalar_value(net_item, "class", "Default")
        nodes = []
        for node_item in _find_children(net_item, "node"):
            ref = _scalar_value(node_item, "ref")
            pin = _scalar_value(node_item, "pin")
            pinfunction = _scalar_value(node_item, "pinfunction")
            pintype = _scalar_value(node_item, "pintype")
            nodes.append({
                "reference": ref,
                "pin": pin,
                "pinfunction": pinfunction,
                "pintype": pintype,
            })
        nets.append({
            "code": int(code) if code.isdigit() else code,
            "name": name,
            "class": net_class,
            "is_unconnected": name.lower().startswith("unconnected-"),
            "node_count": len(nodes),
            "nodes": nodes,
        })

    return {
        "design": design_info,
        "components": netlist_components,
        "nets": nets,
    }


def build_circuit_map(bom_path=DEFAULT_BOM_PATH, netlist_path=DEFAULT_NETLIST_PATH):
    """Link BOM and KiCad netlist, cross-validate references/values, and build pin-to-net connections."""
    bom_path = Path(bom_path).resolve()
    netlist_path = Path(netlist_path).resolve()
    bom_components = parse_bom_csv(bom_path)
    netlist_data = parse_kicad_netlist(netlist_path)
    net_components = netlist_data["components"]
    nets = netlist_data["nets"]

    bom_refs = set(bom_components)
    net_refs = set(net_components)
    matched_refs = sorted(bom_refs & net_refs, key=reference_sort_key)
    bom_only_refs = sorted(bom_refs - net_refs, key=reference_sort_key)
    netlist_only_all = sorted(net_refs - bom_refs, key=reference_sort_key)
    netlist_non_bom_hardware = [
        ref for ref in netlist_only_all if is_non_bom_hardware(net_components[ref])
    ]
    netlist_unexpected_missing_from_bom = [
        ref for ref in netlist_only_all if not is_non_bom_hardware(net_components[ref])
    ]

    value_mismatches = []
    footprint_mismatches = []
    for ref in matched_refs:
        if bom_components[ref]["value"] != net_components[ref]["value"]:
            value_mismatches.append({
                "reference": ref,
                "bom_value": bom_components[ref]["value"],
                "netlist_value": net_components[ref]["value"],
            })
        if bom_components[ref]["footprint"] != net_components[ref]["footprint"]:
            footprint_mismatches.append({
                "reference": ref,
                "bom_footprint": bom_components[ref]["footprint"],
                "netlist_footprint": net_components[ref]["footprint"],
            })

    # Map every component pin to its net and peer connections
    pin_connections_by_ref = {ref: [] for ref in sorted(bom_refs | net_refs, key=reference_sort_key)}
    unknown_node_references = []
    total_pin_nodes = 0

    for net in nets:
        net_name = net["name"]
        net_code = net["code"]
        nodes = net["nodes"]
        for node in nodes:
            total_pin_nodes += 1
            ref = node["reference"]
            if ref not in pin_connections_by_ref:
                unknown_node_references.append({"net": net_name, **node})
                pin_connections_by_ref[ref] = []
            peers = [
                f"{other['reference']}:{other['pin']}"
                for other in nodes
                if not (other["reference"] == ref and other["pin"] == node["pin"])
            ]
            pin_connections_by_ref[ref].append({
                "pin": node["pin"],
                "net_code": net_code,
                "net_name": net_name,
                "pinfunction": node["pinfunction"],
                "pintype": node["pintype"],
                "is_unconnected": net["is_unconnected"],
                "connected_peers": peers,
            })

    all_refs = sorted(bom_refs | net_refs, key=reference_sort_key)
    merged_components = []
    category_counts = {}
    for ref in all_refs:
        bom_entry = bom_components.get(ref, {})
        net_entry = net_components.get(ref, {})
        pins = sorted(pin_connections_by_ref.get(ref, []), key=lambda item: pin_sort_key(item["pin"]))
        active_pins = [p for p in pins if not p["is_unconnected"]]
        unconnected_pins = [p for p in pins if p["is_unconnected"]]
        category = bom_entry.get("category") or net_entry.get("category") or classify_reference(ref)
        category_counts[category] = category_counts.get(category, 0) + 1

        merged_components.append({
            "reference": ref,
            "category": category,
            "value": bom_entry.get("value") or net_entry.get("value", ""),
            "part_number": bom_entry.get("part_number") or net_entry.get("part_number", ""),
            "footprint": bom_entry.get("footprint") or net_entry.get("footprint", ""),
            "datasheet": bom_entry.get("datasheet") or net_entry.get("datasheet", ""),
            "library": net_entry.get("library", ""),
            "library_part": net_entry.get("library_part", ""),
            "sheet_name": net_entry.get("sheet_name", ""),
            "sheet_file": net_entry.get("sheet_file", ""),
            "in_bom": ref in bom_refs,
            "in_netlist": ref in net_refs,
            "pin_count": len(pins),
            "active_pin_count": len(active_pins),
            "unconnected_pin_count": len(unconnected_pins),
            "pins": pins,
        })

    active_nets = [net for net in nets if not net["is_unconnected"]]
    unconnected_nets = [net for net in nets if net["is_unconnected"]]

    is_valid = (
        len(bom_only_refs) == 0
        and len(netlist_unexpected_missing_from_bom) == 0
        and len(value_mismatches) == 0
        and len(footprint_mismatches) == 0
        and len(unknown_node_references) == 0
    )

    return {
        "inputs": {
            "bom_path": str(bom_path),
            "bom_sha256": sha256(bom_path),
            "netlist_path": str(netlist_path),
            "netlist_sha256": sha256(netlist_path),
        },
        "design": netlist_data["design"],
        "validation": {
            "is_valid": is_valid,
            "bom_component_count": len(bom_refs),
            "netlist_component_count": len(net_refs),
            "matched_component_count": len(matched_refs),
            "bom_only_references": bom_only_refs,
            "netlist_only_references": netlist_unexpected_missing_from_bom,
            "netlist_non_bom_hardware": netlist_non_bom_hardware,
            "value_mismatches": value_mismatches,
            "footprint_mismatches": footprint_mismatches,
            "unknown_node_references": unknown_node_references,
            "total_nets": len(nets),
            "active_nets": len(active_nets),
            "unconnected_nets": len(unconnected_nets),
            "total_pin_connections": total_pin_nodes,
        },
        "category_summary": dict(sorted(category_counts.items())),
        "components": merged_components,
        "nets": nets,
    }


def export_circuit_map(circuit_map, output_dir=DEFAULT_OUTPUT_DIR):
    """Write circuit_map.json, components_summary.csv, and pin_connections.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "circuit_map.json"
    json_path.write_text(json.dumps(circuit_map, indent=2) + "\n", encoding="utf-8")

    comp_csv_path = output_dir / "components_summary.csv"
    with comp_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "reference",
                "category",
                "value",
                "part_number",
                "footprint",
                "datasheet",
                "library_part",
                "sheet_file",
                "pin_count",
                "active_pin_count",
                "unconnected_pin_count",
            ],
        )
        writer.writeheader()
        for comp in circuit_map["components"]:
            writer.writerow({k: comp[k] for k in writer.fieldnames})

    pins_csv_path = output_dir / "pin_connections.csv"
    with pins_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "reference",
                "value",
                "part_number",
                "pin",
                "pinfunction",
                "pintype",
                "net_code",
                "net_name",
                "is_unconnected",
                "connected_peers",
            ],
        )
        writer.writeheader()
        for comp in circuit_map["components"]:
            for pin in comp["pins"]:
                writer.writerow({
                    "reference": comp["reference"],
                    "value": comp["value"],
                    "part_number": comp["part_number"],
                    "pin": pin["pin"],
                    "pinfunction": pin["pinfunction"],
                    "pintype": pin["pintype"],
                    "net_code": pin["net_code"],
                    "net_name": pin["net_name"],
                    "is_unconnected": pin["is_unconnected"],
                    "connected_peers": ";".join(pin["connected_peers"]),
                })

    return {
        "circuit_map_json": json_path,
        "components_csv": comp_csv_path,
        "pin_connections_csv": pins_csv_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bom", type=Path, default=DEFAULT_BOM_PATH, help="Path to BOM CSV file")
    parser.add_argument("--netlist", type=Path, default=DEFAULT_NETLIST_PATH, help="Path to KiCad .net file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for exported artifacts")
    parser.add_argument("--ref", help="Inspect a specific component reference (e.g., Q1, U3, U5)")
    args = parser.parse_args()

    circuit_map = build_circuit_map(args.bom, args.netlist)
    artifacts = export_circuit_map(circuit_map, args.output_dir)
    val = circuit_map["validation"]

    print(
        f"Circuit Map Validation: {'PASSED' if val['is_valid'] else 'FAILED'} | "
        f"BOM Components: {val['bom_component_count']} | "
        f"Netlist Components: {val['netlist_component_count']} | "
        f"Matched: {val['matched_component_count']}"
    )
    print(
        f"Nets: {val['total_nets']} total ({val['active_nets']} active, "
        f"{val['unconnected_nets']} unconnected) | "
        f"Total Pin Nodes: {val['total_pin_connections']}"
    )

    if args.ref:
        target = args.ref.strip().upper()
        match = next((c for c in circuit_map["components"] if c["reference"].upper() == target), None)
        if match is None:
            raise SystemExit(f"Component reference '{args.ref}' not found in circuit map.")
        print(f"\nComponent {match['reference']} ({match['category']}):")
        print(f"  Value / Part Number: {match['value']} (Part: {match['part_number']})")
        print(f"  Footprint:           {match['footprint']}")
        print(f"  Datasheet:           {match['datasheet'] or 'N/A'}")
        print(f"  Pins ({match['pin_count']} total, {match['active_pin_count']} active):")
        for pin in match["pins"]:
            peers = ", ".join(pin["connected_peers"]) if pin["connected_peers"] else "none"
            fn = f" ({pin['pinfunction']})" if pin["pinfunction"] else ""
            print(f"    Pin {pin['pin']}{fn} [{pin['pintype']}] -> Net '{pin['net_name']}' | Peers: {peers}")

    print(f"\nExported circuit map JSON: {artifacts['circuit_map_json']}")
    print(f"Exported components CSV:   {artifacts['components_csv']}")
    print(f"Exported pin connections:  {artifacts['pin_connections_csv']}")


if __name__ == "__main__":
    main()
