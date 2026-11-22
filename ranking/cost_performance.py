"""Module 4: Cost and performance comparison for eligible MOSFET replacements from Module 3.

Ranks eligible candidates by normalized unit price (matched quantity and currency), stock
availability, and electrical performance (R_DS(on), current/voltage headroom). Checks supplier
quote freshness timestamps and explains engineering/purchasing trade-offs without guaranteeing
circuit compatibility or improved circuit performance.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from components.circuit_map import DEFAULT_BOM_PATH, DEFAULT_NETLIST_PATH
from recommendations.recommender import DEFAULT_CATALOG_PATH, recommend_replacements

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUOTES_PATH = PROJECT_ROOT / "data/inputs/supplier_quotes.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/ranking"
DEFAULT_REFERENCE_TIME = "2026-11-22T00:00:00Z"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_iso_timestamp(ts_str):
    text = str(ts_str).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_supplier_quotes(quotes_path=DEFAULT_QUOTES_PATH):
    """Load and validate timestamped supplier quotes."""
    quotes_path = Path(quotes_path)
    if not quotes_path.is_file():
        raise FileNotFoundError(f"Missing supplier quotes file: {quotes_path}")

    payload = json.loads(quotes_path.read_text(encoding="utf-8"))
    quotes = payload.get("quotes")
    rates = payload.get("exchange_rates_to_usd")
    if not isinstance(quotes, list) or not quotes:
        raise ValueError(f"Quotes file {quotes_path} must contain a non-empty 'quotes' list")
    if not isinstance(rates, dict) or "USD" not in rates:
        raise ValueError(f"Quotes file {quotes_path} must define 'exchange_rates_to_usd'")

    for idx, q in enumerate(quotes, 1):
        for field in ("part_number", "supplier", "currency", "stock", "quote_timestamp", "price_breaks"):
            if field not in q:
                raise ValueError(f"Quote #{idx} missing required field '{field}'")
        if q["currency"] not in rates:
            raise ValueError(f"Quote #{idx} uses unsupported currency '{q['currency']}'")
        if not isinstance(q["price_breaks"], list) or not q["price_breaks"]:
            raise ValueError(f"Quote #{idx} must have a non-empty 'price_breaks' list")
        parse_iso_timestamp(q["quote_timestamp"])

    return payload


def unit_price_at_quantity(price_breaks, quantity):
    """Find the active unit price break for the requested purchase quantity."""
    if quantity < 1:
        raise ValueError("quantity must be at least 1")
    ordered = sorted(price_breaks, key=lambda b: int(b["min_qty"]))
    selected = ordered[0]["unit_price"]
    for tier in ordered:
        if quantity >= int(tier["min_qty"]):
            selected = tier["unit_price"]
    return float(selected)


def convert_currency(amount, from_currency, to_currency, rates_to_usd):
    """Convert price between currencies using fixed snapshot exchange rates to USD."""
    if from_currency not in rates_to_usd:
        raise ValueError(f"Unknown source currency: {from_currency}")
    if to_currency not in rates_to_usd:
        raise ValueError(f"Unknown target currency: {to_currency}")
    usd_amount = float(amount) * float(rates_to_usd[from_currency])
    return usd_amount / float(rates_to_usd[to_currency])


def select_best_quote_for_part(
    part_number,
    quotes_payload,
    quantity=100,
    currency="USD",
    reference_time=DEFAULT_REFERENCE_TIME,
    max_freshness_days=None,
):
    """Select the lowest normalized-price quote with sufficient stock for a candidate part."""
    rates = quotes_payload["exchange_rates_to_usd"]
    max_age = (
        max_freshness_days
        if max_freshness_days is not None
        else int(quotes_payload.get("max_freshness_days", 30))
    )
    ref_dt = parse_iso_timestamp(reference_time)

    part_quotes = [
        q for q in quotes_payload["quotes"]
        if q["part_number"].upper() == part_number.upper()
    ]
    if not part_quotes:
        return None

    evaluated_quotes = []
    for q in part_quotes:
        raw_unit = unit_price_at_quantity(q["price_breaks"], quantity)
        norm_unit = round(convert_currency(raw_unit, q["currency"], currency, rates), 4)
        total_cost = round(norm_unit * quantity, 2)
        quote_dt = parse_iso_timestamp(q["quote_timestamp"])
        age_days = round(max(0.0, (ref_dt - quote_dt).total_seconds() / 86400.0), 1)
        is_stale = age_days > max_age
        sufficient_stock = int(q["stock"]) >= quantity

        evaluated_quotes.append({
            "supplier": q["supplier"],
            "sku": q.get("sku", ""),
            "original_currency": q["currency"],
            "original_unit_price": raw_unit,
            "normalized_currency": currency,
            "normalized_unit_price": norm_unit,
            "quantity": quantity,
            "total_price": total_cost,
            "stock": int(q["stock"]),
            "sufficient_stock": sufficient_stock,
            "lead_time_days": int(q.get("lead_time_days", 0)),
            "quote_timestamp": q["quote_timestamp"],
            "quote_age_days": age_days,
            "is_stale_quote": is_stale,
        })

    # Prefer in-stock quotes first, then lowest normalized unit price, then fresher quote
    evaluated_quotes.sort(
        key=lambda item: (
            not item["sufficient_stock"],
            item["normalized_unit_price"],
            item["quote_age_days"],
        )
    )
    best = dict(evaluated_quotes[0])
    best["all_supplier_quotes"] = [dict(q) for q in evaluated_quotes]
    return best


def explain_tradeoffs(entry, cheapest_price, best_rds_on, highest_id):
    """Build explicit engineering and purchasing trade-off explanations for human review."""
    notes = []
    quote = entry["best_quote"]
    specs = entry["specs"]
    price = quote["normalized_unit_price"]
    currency = quote["normalized_currency"]

    if abs(price - cheapest_price) < 1e-6:
        notes.append(f"Lowest unit cost ({price:.4f} {currency} @ qty {quote['quantity']})")
    else:
        diff_pct = ((price - cheapest_price) / max(cheapest_price, 1e-6)) * 100.0
        notes.append(
            f"+{diff_pct:.1f}% higher cost than cheapest option ({price:.4f} vs {cheapest_price:.4f} {currency})"
        )

    rds = specs.get("rds_on_max_ohm")
    if rds is not None:
        if abs(rds - best_rds_on) < 1e-6:
            notes.append(f"Lowest conduction loss (R_DS(on) = {rds} Ohm)")
        else:
            notes.append(f"Higher on-resistance (R_DS(on) = {rds} Ohm vs best {best_rds_on} Ohm)")

    id_max = specs.get("id_max_a")
    if id_max is not None and abs(id_max - highest_id) < 1e-6:
        notes.append(f"Highest current capability (I_D = {id_max}A)")

    if quote["is_stale_quote"]:
        notes.append(
            f"STALE QUOTE ALERT: quote is {quote['quote_age_days']} days old ({quote['quote_timestamp']}); refresh supplier pricing"
        )
    if not quote["sufficient_stock"]:
        notes.append(
            f"LOW STOCK ALERT: supplier stock ({quote['stock']}) is below requested quantity ({quote['quantity']})"
        )
    if entry["requires_human_review"]:
        missing = entry["missing_required_specs"] + entry["missing_optional_specs"]
        notes.append(
            f"DATASHEET REVIEW REQUIRED: catalog record omits {sorted(set(missing))}"
        )

    return "; ".join(notes)


def rank_eligible_candidates(
    reference="Q1",
    quantity=100,
    currency="USD",
    weight_cost=0.45,
    weight_performance=0.35,
    weight_stock=0.20,
    recommendations_report=None,
    quotes_path=DEFAULT_QUOTES_PATH,
    catalog_path=DEFAULT_CATALOG_PATH,
    bom_path=DEFAULT_BOM_PATH,
    netlist_path=DEFAULT_NETLIST_PATH,
    reference_time=DEFAULT_REFERENCE_TIME,
    max_freshness_days=None,
    **recommender_kwargs,
):
    """Rank eligible MOSFET candidates from Module 3 by normalized price, stock, and performance."""
    if quantity < 1:
        raise ValueError("quantity must be at least 1")
    total_weight = weight_cost + weight_performance + weight_stock
    if total_weight <= 0:
        raise ValueError("Ranking weights must sum to a positive number")
    w_cost = weight_cost / total_weight
    w_perf = weight_performance / total_weight
    w_stock = weight_stock / total_weight

    if recommendations_report is None:
        recommendations_report = recommend_replacements(
            reference=reference,
            catalog_path=catalog_path,
            bom_path=bom_path,
            netlist_path=netlist_path,
            **recommender_kwargs,
        )
    quotes_payload = load_supplier_quotes(quotes_path)
    currency = currency.strip().upper()

    eligible = recommendations_report["eligible_candidates"]
    paired = []
    unquoted_parts = []
    for candidate in eligible:
        best_quote = select_best_quote_for_part(
            part_number=candidate["part_number"],
            quotes_payload=quotes_payload,
            quantity=quantity,
            currency=currency,
            reference_time=reference_time,
            max_freshness_days=max_freshness_days,
        )
        if best_quote is None:
            unquoted_parts.append(candidate["part_number"])
            continue
        paired.append({**candidate, "best_quote": best_quote})

    if not paired:
        return {
            "target_component": recommendations_report["target_component"],
            "purchasing_conditions": {
                "quantity": quantity,
                "currency": currency,
                "reference_time": reference_time,
            },
            "weights": {"cost": round(w_cost, 3), "performance": round(w_perf, 3), "stock": round(w_stock, 3)},
            "ranked_candidates": [],
            "unquoted_candidates": unquoted_parts,
            "alerts": {"stale_quote_parts": [], "insufficient_stock_parts": [], "missing_spec_parts": []},
            "disclaimer": (
                "Rankings compare normalized price, stock, and datasheet limits under matched quantity "
                "and currency. They do not guarantee circuit compatibility or improved performance."
            ),
        }

    prices = [item["best_quote"]["normalized_unit_price"] for item in paired]
    min_price, max_price = min(prices), max(prices)

    perf_raw = []
    for item in paired:
        specs = item["specs"]
        rds = specs.get("rds_on_max_ohm") or 15.0
        id_max = specs.get("id_max_a") or 0.1
        vds_max = specs.get("vds_max_v") or 25.0
        # Lower R_DS(on) and higher I_D / V_DS headroom yield higher electrical performance
        raw = (1.0 / max(rds, 0.05)) * 50.0 + id_max * 30.0 + (vds_max / 60.0) * 10.0
        perf_raw.append(raw)
    min_perf, max_perf = min(perf_raw), max(perf_raw)

    stocks = [item["best_quote"]["stock"] for item in paired]
    max_stock = max(stocks) if stocks else 1

    best_rds_on = min(
        (item["specs"]["rds_on_max_ohm"] for item in paired if item["specs"].get("rds_on_max_ohm") is not None),
        default=0.0,
    )
    highest_id = max(
        (item["specs"]["id_max_a"] for item in paired if item["specs"].get("id_max_a") is not None),
        default=0.0,
    )

    ranked = []
    stale_parts = []
    low_stock_parts = []
    missing_spec_parts = []

    for item, raw_p in zip(paired, perf_raw):
        quote = item["best_quote"]
        price = quote["normalized_unit_price"]

        # Normalize each sub-score to [0, 100]
        if max_price > min_price:
            cost_score = ((max_price - price) / (max_price - min_price)) * 100.0
        else:
            cost_score = 100.0

        if max_perf > min_perf:
            performance_score = ((raw_p - min_perf) / (max_perf - min_perf)) * 100.0
        else:
            performance_score = 100.0

        stock_ratio = min(1.0, quote["stock"] / max(quantity * 10.0, float(max_stock) * 0.5, 1.0))
        stock_score = (stock_ratio * 100.0) if quote["sufficient_stock"] else (stock_ratio * 35.0)

        # Penalize stale quotes or missing datasheet specs slightly so complete/fresh parts win ties
        penalty = 0.0
        if quote["is_stale_quote"]:
            penalty += 5.0
            stale_parts.append(item["part_number"])
        if not quote["sufficient_stock"]:
            penalty += 15.0
            low_stock_parts.append(item["part_number"])
        if item["requires_human_review"]:
            penalty += 5.0
            missing_spec_parts.append(item["part_number"])

        composite = max(
            0.0,
            w_cost * cost_score + w_perf * performance_score + w_stock * stock_score - penalty,
        )

        entry = {
            "part_number": item["part_number"],
            "manufacturer": item["manufacturer"],
            "channel": item["channel"],
            "package": item["package"],
            "pinout": item["pinout"],
            "specs": item["specs"],
            "requires_human_review": item["requires_human_review"],
            "missing_required_specs": item["missing_required_specs"],
            "missing_optional_specs": item["missing_optional_specs"],
            "best_quote": quote,
            "scores": {
                "cost_score": round(cost_score, 2),
                "performance_score": round(performance_score, 2),
                "stock_score": round(stock_score, 2),
                "penalty": round(penalty, 2),
                "total_score": round(composite, 2),
            },
            "price_vs_cheapest_diff": round(price - min_price, 4),
            "total_order_diff_vs_cheapest": round((price - min_price) * quantity, 2),
            "datasheet": item["datasheet"],
        }
        entry["tradeoff_explanation"] = explain_tradeoffs(entry, min_price, best_rds_on, highest_id)
        ranked.append(entry)

    ranked.sort(
        key=lambda r: (
            -r["scores"]["total_score"],
            r["best_quote"]["normalized_unit_price"],
            r["part_number"],
        )
    )
    for rank_idx, entry in enumerate(ranked, 1):
        entry["rank"] = rank_idx

    return {
        "quotes_version": quotes_payload.get("quotes_version", "unknown"),
        "quotes_path": str(Path(quotes_path).resolve()),
        "quotes_sha256": sha256(quotes_path),
        "target_component": recommendations_report["target_component"],
        "operating_requirements": recommendations_report["operating_requirements"],
        "purchasing_conditions": {
            "quantity": quantity,
            "currency": currency,
            "reference_time": reference_time,
            "max_freshness_days": (
                max_freshness_days
                if max_freshness_days is not None
                else int(quotes_payload.get("max_freshness_days", 30))
            ),
        },
        "weights": {
            "cost": round(w_cost, 3),
            "performance": round(w_perf, 3),
            "stock": round(w_stock, 3),
        },
        "price_summary": {
            "currency": currency,
            "quantity": quantity,
            "lowest_unit_price": round(min_price, 4),
            "highest_unit_price": round(max_price, 4),
            "max_total_savings_at_quantity": round((max_price - min_price) * quantity, 2),
        },
        "alerts": {
            "stale_quote_parts": stale_parts,
            "insufficient_stock_parts": low_stock_parts,
            "missing_spec_parts": missing_spec_parts,
        },
        "ranked_candidates": ranked,
        "unquoted_candidates": unquoted_parts,
        "disclaimer": (
            "Weighted ranking compares eligible candidates under matched purchase quantity and "
            "currency. Trade-off explanations support human engineering and procurement review "
            "without guaranteeing circuit compatibility or improved circuit performance."
        ),
    }


def export_ranking(report, output_dir=DEFAULT_OUTPUT_DIR):
    """Export Module 4 cost and performance comparison report to JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = report["target_component"]["reference"].lower()
    qty = report["purchasing_conditions"]["quantity"]
    curr = report["purchasing_conditions"]["currency"].lower()

    json_path = output_dir / f"ranking_{ref}_qty{qty}_{curr}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / f"ranking_{ref}_qty{qty}_{curr}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "part_number",
                "manufacturer",
                "supplier",
                "sku",
                "quantity",
                "currency",
                "unit_price",
                "total_price",
                "price_vs_cheapest_diff",
                "stock",
                "lead_time_days",
                "quote_timestamp",
                "quote_age_days",
                "is_stale_quote",
                "rds_on_max_ohm",
                "id_max_a",
                "vds_max_v",
                "cost_score",
                "performance_score",
                "stock_score",
                "total_score",
                "requires_human_review",
                "tradeoff_explanation",
            ],
        )
        writer.writeheader()
        for item in report["ranked_candidates"]:
            q = item["best_quote"]
            s = item["scores"]
            specs = item["specs"]
            writer.writerow({
                "rank": item["rank"],
                "part_number": item["part_number"],
                "manufacturer": item["manufacturer"],
                "supplier": q["supplier"],
                "sku": q["sku"],
                "quantity": q["quantity"],
                "currency": q["normalized_currency"],
                "unit_price": q["normalized_unit_price"],
                "total_price": q["total_price"],
                "price_vs_cheapest_diff": item["price_vs_cheapest_diff"],
                "stock": q["stock"],
                "lead_time_days": q["lead_time_days"],
                "quote_timestamp": q["quote_timestamp"],
                "quote_age_days": q["quote_age_days"],
                "is_stale_quote": q["is_stale_quote"],
                "rds_on_max_ohm": specs.get("rds_on_max_ohm"),
                "id_max_a": specs.get("id_max_a"),
                "vds_max_v": specs.get("vds_max_v"),
                "cost_score": s["cost_score"],
                "performance_score": s["performance_score"],
                "stock_score": s["stock_score"],
                "total_score": s["total_score"],
                "requires_human_review": item["requires_human_review"],
                "tradeoff_explanation": item["tradeoff_explanation"],
            })

    return {"json": json_path, "csv": csv_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="Q1", help="Target component reference (e.g., Q1, Q2, Q3)")
    parser.add_argument("--quantity", type=int, default=100, help="Order quantity for price tier matching")
    parser.add_argument("--currency", default="USD", choices=["USD", "EUR", "GBP"], help="Normalized comparison currency")
    parser.add_argument("--weight-cost", type=float, default=0.45)
    parser.add_argument("--weight-performance", type=float, default=0.35)
    parser.add_argument("--weight-stock", type=float, default=0.20)
    parser.add_argument("--quotes", type=Path, default=DEFAULT_QUOTES_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--bom", type=Path, default=DEFAULT_BOM_PATH)
    parser.add_argument("--netlist", type=Path, default=DEFAULT_NETLIST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = rank_eligible_candidates(
        reference=args.ref,
        quantity=args.quantity,
        currency=args.currency,
        weight_cost=args.weight_cost,
        weight_performance=args.weight_performance,
        weight_stock=args.weight_stock,
        quotes_path=args.quotes,
        catalog_path=args.catalog,
        bom_path=args.bom,
        netlist_path=args.netlist,
    )
    artifacts = export_ranking(report, args.output_dir)

    cond = report["purchasing_conditions"]
    ps = report["price_summary"]
    print(
        f"Target: {report['target_component']['reference']} | "
        f"Matched Purchasing Conditions: Qty={cond['quantity']}, Currency={cond['currency']} | "
        f"Weights: {report['weights']}"
    )
    print(
        f"Unit Price Range: {ps['lowest_unit_price']:.4f} - {ps['highest_unit_price']:.4f} {ps['currency']} "
        f"(Max savings at Qty {ps['quantity']}: {ps['max_total_savings_at_quantity']:.2f} {ps['currency']})"
    )
    if any(report["alerts"].values()):
        print(f"Alerts: {report['alerts']}")

    print("\nRanked Eligible Candidates:")
    for item in report["ranked_candidates"]:
        q = item["best_quote"]
        s = item["scores"]
        specs = item["specs"]
        print(
            f"  #{item['rank']} {item['part_number']} ({item['manufacturer']}) | "
            f"Total Score: {s['total_score']:.2f} "
            f"[Cost={s['cost_score']:.1f}, Perf={s['performance_score']:.1f}, Stock={s['stock_score']:.1f}]\n"
            f"     Best Quote: {q['supplier']} ({q['sku']}) -> {q['normalized_unit_price']:.4f} {q['normalized_currency']}/unit "
            f"(Total: {q['total_price']:.2f} {q['normalized_currency']} @ Qty {q['quantity']}) | "
            f"Stock: {q['stock']} | Timestamp: {q['quote_timestamp']} ({q['quote_age_days']}d old)\n"
            f"     Electrical: R_DS(on)={specs['rds_on_max_ohm']} Ohm, I_D={specs['id_max_a']}A, V_DS={specs['vds_max_v']}V\n"
            f"     Trade-offs: {item['tradeoff_explanation']}"
        )

    print(f"\nExported JSON report: {artifacts['json']}")
    print(f"Exported CSV table:   {artifacts['csv']}")


if __name__ == "__main__":
    main()
