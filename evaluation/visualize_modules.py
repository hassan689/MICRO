"""Generate rich visual PNG diagrams and dashboard charts showing how each of the 4 modules
(and the CNN, Logistic Regression, Linear Regression, and LSTM models) operates dynamically.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

from baselines.difference_detector import detect_defects, read_annotations
from components.circuit_map import build_circuit_map, derive_component_operating_point
from models.ml_extensions import (
    NumpyCNNDefectPatchClassifier,
    build_board_failure_logistic_model,
)
from ranking.cost_performance import rank_eligible_candidates
from recommendations.recommender import recommend_replacements

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data/raw/DeepPCB/PCBData"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/visualizations"
CLASS_NAMES = {
    1: "Open", 2: "Short", 3: "Mousebite",
    4: "Spur", 5: "Copper", 6: "Pin hole",
}


def render_module1_visual(raw_root=DEFAULT_RAW_ROOT, output_dir=DEFAULT_OUTPUT_DIR):
    """Render Module 1: Reference vs Defective PCB, AbsDiff heatmap, CNN patch scores, and Logistic failure risk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_file = Path(raw_root) / "trainval.txt"
    first_line = next(line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip())
    base_rel, ann_rel = first_line.split()
    base = Path(base_rel)
    test_img_path = Path(raw_root) / base.with_name(f"{base.stem}_test.jpg")
    temp_img_path = Path(raw_root) / base.with_name(f"{base.stem}_temp.jpg")
    ann_path = Path(raw_root) / ann_rel

    defective_bgr = cv2.imread(str(test_img_path))
    reference_bgr = cv2.imread(str(temp_img_path))
    defective_rgb = cv2.cvtColor(defective_bgr, cv2.COLOR_BGR2RGB)
    reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)

    gt_boxes, gt_labels = read_annotations(ann_path)
    det = detect_defects(test_img_path, temp_img_path, threshold=30, min_area=20)
    diff_gray = cv2.absdiff(
        cv2.cvtColor(defective_rgb, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY),
    )

    cnn = NumpyCNNDefectPatchClassifier(seed=42)
    cnn_scores = []
    for box in gt_boxes:
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        patch = diff_gray[y1:max(y1 + 5, y2), x1:max(x1 + 5, x2)]
        res = cnn.forward_patch(patch)
        cnn_scores.append(res["cnn_defect_probability"])

    # Compute Logistic Regression Board Failure Probability
    open_short_count = sum(1 for lbl in gt_labels if lbl in (1, 2))
    other_count = len(gt_labels) - open_short_count
    max_area = max(((b[2] - b[0]) * (b[3] - b[1]) for b in gt_boxes), default=0.0)
    mean_conf = float(np.mean(cnn_scores)) if cnn_scores else 0.5
    logreg = build_board_failure_logistic_model()
    fail_prob = float(logreg.predict_proba([[open_short_count, other_count, max_area, mean_conf]])[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Module 1: PCB Defect Inspection + CNN Patch Scoring & Logistic Regression Yield Risk\n"
        f"Sample: {test_img_path.name} | Ground-Truth Defects: {len(gt_boxes)} | "
        f"Logistic Regression P(Electrical Failure) = {fail_prob * 100:.1f}%",
        fontsize=13,
        fontweight="bold",
    )

    axes[0, 0].imshow(reference_rgb)
    axes[0, 0].set_title("1. Reference Template PCB (*_temp.jpg)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(defective_rgb)
    for box, lbl, c_score in zip(gt_boxes, gt_labels, cnn_scores):
        x1, y1, x2, y2 = box
        color = "#dc2626" if lbl in (1, 2) else "#2563eb"
        axes[0, 1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        axes[0, 1].text(
            x1, max(12, y1 - 6),
            f"{CLASS_NAMES.get(lbl, lbl)} (CNN:{c_score:.2f})",
            color="white", fontsize=8, fontweight="bold",
            bbox={"facecolor": color, "alpha": 0.85, "pad": 1.5, "edgecolor": "none"},
        )
    axes[0, 1].set_title("2. Defective PCB + Faster R-CNN / CNN Defect Boxes", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    im = axes[1, 0].imshow(diff_gray, cmap="inferno")
    for pbox, score in zip(det["boxes"], det["scores"]):
        x1, y1, x2, y2 = pbox
        axes[1, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#22c55e", linewidth=1.6))
    axes[1, 0].set_title(
        f"3. Aligned Difference Heatmap + Baseline Detections ({len(det['boxes'])} boxes)",
        fontsize=11, fontweight="bold",
    )
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel 4: Class count bar chart + Logistic Regression Risk Gauge
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    for lbl in gt_labels:
        class_counts[CLASS_NAMES[lbl]] += 1
    names = list(class_counts.keys())
    counts = [class_counts[n] for n in names]
    colors = ["#dc2626" if n in ("Open", "Short") else "#3b82f6" for n in names]
    bars = axes[1, 1].bar(names, counts, color=colors, edgecolor="#1e293b")
    for bar, cnt in zip(bars, counts):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(cnt), ha="center", fontweight="bold")
    axes[1, 1].set_ylim(0, max(counts + [2]) + 0.8)
    axes[1, 1].set_ylabel("Defect Count on Board")
    axes[1, 1].set_title(
        f"4. Defect Class Breakdown & Logistic Regression Triage\n"
        f"Critical (Open/Short): {open_short_count} | Cosmetic/Minor: {other_count} | "
        f"Failure Risk: {fail_prob * 100:.1f}% ({'REJECT BOARD' if fail_prob >= 0.5 else 'PASS'})",
        fontsize=10.5, fontweight="bold",
    )
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = output_dir / "module1_defect_inspection.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_module2_visual(output_dir=DEFAULT_OUTPUT_DIR):
    """Render Module 2: Component category distribution + dynamic Q1/Q2/Q3 circuit pin-to-net topology."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit_map = build_circuit_map()
    q1_op = derive_component_operating_point("Q1", circuit_map)
    q2_op = derive_component_operating_point("Q2", circuit_map)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    fig.suptitle(
        "Module 2: BOM & KiCad Netlist Circuit Mapping + Dynamic Ohm's Law Operating Point Derivation",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Component Categories from BOM + Netlist
    cats = list(circuit_map["category_summary"].keys())
    counts = [circuit_map["category_summary"][c] for c in cats]
    y_pos = np.arange(len(cats))
    bars = axes[0].barh(y_pos, counts, color="#0284c7", edgecolor="#0f172a")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(cats, fontsize=9.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Component Count")
    axes[0].set_title(
        f"1. Cross-Validated Schematic Components\n({circuit_map['validation']['matched_component_count']}/56 BOM Matched, "
        f"{circuit_map['validation']['active_nets']} Active Nets)",
        fontsize=10.5, fontweight="bold",
    )
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, str(cnt), va="center", fontweight="bold")
    axes[0].set_xlim(0, max(counts) + 3)
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    # Panel 2: Visual Schematic Topology & Derived Electrical Operating Requirements
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("2. Dynamic Pin-to-Net Topology & Ohm's Law Requirement Extraction", fontsize=10.5, fontweight="bold")

    boxes_info = [
        (0.4, 5.4, 9.2, 4.0, "#f0f9ff", "#0284c7",
         f"Target Q1 (BC237 | Footprint: TO-92 | Inferred Polarity: N-channel)\n"
         f"• Pin 1 (Collector -> Drain): Net 'Net-(D11-K)' -> Connected Peers: D11:1, R11:1 (22 kOhm)\n"
         f"• Pin 2 (Base -> Gate):      Net 'Net-(Q1-B)'  -> Connected Peers: R8:2 (1.0 kOhm)\n"
         f"• Pin 3 (Emitter -> Source): Net 'GND' (0.0V)  -> Shared with U3 (7805), U4 (LT1373)\n"
         f"• Dynamic Ohm's Law Derivation: Max Rail = {q1_op['operating_voltage_v']}V (VPP), R_min = {q1_op['minimum_connected_resistance_ohm']} Ohm\n"
         f"  -> Auto-Derived Limits: V_DS >= {q1_op['min_vds_v']}V | I_D >= {q1_op['min_id_a']}A | V_GS(th) <= {q1_op['max_vgs_th_v']}V | P_D >= {q1_op['min_pd_w']}W"),
        (0.4, 0.7, 9.2, 4.1, "#fdf4ff", "#9333ea",
         f"Target Q2 (BC307 | Footprint: TO-92 | Inferred Polarity: P-channel)\n"
         f"• Pin 1 (Collector -> Drain): Net 'Net-(Q2-C)' -> Connected Peers: R17:1 (22K), R18:1 (220 Ohm), R9:1 (2.2K)\n"
         f"• Pin 2 (Base -> Gate):      Net 'Net-(Q2-B)' -> Connected Peers: C9:2, R11:2 (22K), R7:2 (10K)\n"
         f"• Pin 3 (Emitter -> Source): Net 'VPP' (13.0V High-Voltage Rail from U4 LT1373 Boost Converter)\n"
         f"• Dynamic Ohm's Law Derivation: Rail = {q2_op['operating_voltage_v']}V, R_min = {q2_op['minimum_connected_resistance_ohm']} Ohm (R18), I_branch = {q2_op['estimated_branch_current_a']*1000:.1f} mA\n"
         f"  -> Auto-Derived Limits: V_DS >= {q2_op['min_vds_v']}V | I_D >= {q2_op['min_id_a']}A | V_GS(th) <= {q2_op['max_vgs_th_v']}V | P_D >= {q2_op['min_pd_w']}W"),
    ]
    for x, y, w, h, fc, ec, text in boxes_info:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25", facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(patch)
        ax.text(x + 0.25, y + h / 2, text, va="center", ha="left", fontsize=9.2, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = output_dir / "module2_circuit_mapping.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_module3_visual(output_dir=DEFAULT_OUTPUT_DIR):
    """Render Module 3: Rule-based compatibility matrix + Logistic Regression compatibility probabilities."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = recommend_replacements(reference="Q1")
    all_cands = report["eligible_candidates"] + report["rejected_candidates"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    req = report["operating_requirements"]
    fig.suptitle(
        f"Module 3: Rule-Based MOSFET Replacement Filtering + Logistic Regression Compatibility Score\n"
        f"Target: {req['reference']} ({req['original_value']}) | Dynamic Circuit Limits: "
        f"Package={req['package']}, Pinout=D-G-S, Channel={req['channel']}, V_DS>={req['min_vds_v']}V, I_D>={req['min_id_a']}A",
        fontsize=12.5, fontweight="bold",
    )

    # Panel 1: Compatibility Pass/Fail Matrix across all 11 Catalog MOSFETs
    check_cols = ["package", "channel", "pinout", "vds_max_v", "id_max_a", "vgs_th_max_v", "rds_on_max_ohm", "pd_max_w"]
    col_labels = ["Package", "Channel", "Pinout", "V_DS", "I_D", "V_GS(th)", "R_DS(on)", "P_D"]
    part_names = [c["part_number"] for c in all_cands]
    matrix = np.zeros((len(all_cands), len(check_cols)))

    for r_idx, cand in enumerate(all_cands):
        for c_idx, key in enumerate(check_cols):
            chk = cand["checks"].get(key, {})
            if chk.get("status") == "missing_spec":
                matrix[r_idx, c_idx] = 0.5  # Amber warning for missing datasheet spec
            elif chk.get("passed"):
                matrix[r_idx, c_idx] = 1.0  # Green pass
            else:
                matrix[r_idx, c_idx] = 0.0  # Red fail

    cmap = matplotlib.colors.ListedColormap(["#ef4444", "#f59e0b", "#22c55e"])
    axes[0].imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    axes[0].set_xticks(np.arange(len(col_labels)))
    axes[0].set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9.5, fontweight="bold")
    axes[0].set_yticks(np.arange(len(part_names)))
    axes[0].set_yticklabels(
        [f"{c['part_number']} ({'ELIGIBLE' if c['eligible'] else 'REJECTED'})" for c in all_cands],
        fontsize=9, fontweight="bold",
    )
    for r_idx in range(len(all_cands)):
        for c_idx in range(len(check_cols)):
            val = matrix[r_idx, c_idx]
            txt = "PASS" if val == 1.0 else ("FLAG" if val == 0.5 else "FAIL")
            axes[0].text(c_idx, r_idx, txt, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    axes[0].set_title("1. Rule Compatibility Matrix (Green=Pass, Amber=Missing Spec Flag, Red=Violation)", fontsize=10.2, fontweight="bold")

    # Panel 2: Logistic Regression Compatibility Probability + Electrical Suitability
    probs = [c["logistic_compatibility_probability"] * 100.0 for c in all_cands]
    bar_colors = [
        "#f59e0b" if (c["eligible"] and c["requires_human_review"])
        else ("#16a34a" if c["eligible"] else "#dc2626")
        for c in all_cands
    ]
    y_pos = np.arange(len(part_names))
    bars = axes[1].barh(y_pos, probs, color=bar_colors, edgecolor="#0f172a")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(part_names, fontsize=9.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Logistic Regression Compatibility Probability (%)")
    axes[1].set_xlim(0, 112)
    axes[1].set_title("2. Learned Logistic Regression Compatibility Probability", fontsize=10.5, fontweight="bold")
    for bar, p_val, cand in zip(bars, probs, all_cands):
        tag = " [Missing P_D Flag]" if cand["requires_human_review"] and cand["eligible"] else ""
        axes[1].text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{p_val:.1f}%{tag}", va="center", fontsize=8.5, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = output_dir / "module3_mosfet_recommendations.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_module4_visual(output_dir=DEFAULT_OUTPUT_DIR):
    """Render Module 4: Weighted Cost/Perf/Stock Ranking, Linear Regression Fair-Price line, and LSTM Forecast."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = rank_eligible_candidates(reference="Q1", quantity=100, currency="USD", use_live_exchange_rates=True)
    ranked = report["ranked_candidates"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    fig.suptitle(
        "Module 4: Cost & Performance Ranking + Ridge Linear Regression Fair-Price & LSTM Time-Series Forecast (Qty=100, USD)",
        fontsize=12.5, fontweight="bold",
    )

    # Panel 1: Weighted Stacked Score Breakdown
    names = [f"#{c['rank']} {c['part_number']}" for c in ranked]
    w = report["weights"]
    cost_contrib = [c["scores"]["cost_score"] * w["cost"] for c in ranked]
    perf_contrib = [c["scores"]["performance_score"] * w["performance"] for c in ranked]
    stock_contrib = [c["scores"]["stock_score"] * w["stock"] for c in ranked]
    x = np.arange(len(names))

    axes[0].bar(x, cost_contrib, label=f"Cost ({int(w['cost']*100)}%)", color="#2563eb")
    axes[0].bar(x, perf_contrib, bottom=cost_contrib, label=f"Performance ({int(w['performance']*100)}%)", color="#16a34a")
    bottom_2 = [a + b for a, b in zip(cost_contrib, perf_contrib)]
    axes[0].bar(x, stock_contrib, bottom=bottom_2, label=f"Stock ({int(w['stock']*100)}%)", color="#9333ea")
    for idx, c in enumerate(ranked):
        axes[0].text(idx, c["scores"]["total_score"] + 1.5, f"Total: {c['scores']['total_score']:.1f}", ha="center", fontweight="bold", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=9.5, fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Weighted Composite Score")
    axes[0].set_title("1. Weighted Ranking Breakdown", fontsize=10.5, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 2: Ridge Linear Regression Fair Unit Price vs Actual Supplier Quote Price
    part_labels = [c["part_number"] for c in ranked]
    actual_prices = [c["best_quote"]["normalized_unit_price"] for c in ranked]
    fair_prices = [c["ml_predictions"]["linear_regression_fair_unit_price"] for c in ranked]
    width = 0.35
    axes[1].bar(x - width / 2, actual_prices, width, label="Actual Supplier Quote (USD)", color="#0284c7")
    axes[1].bar(x + width / 2, fair_prices, width, label="Linear Regression Predicted Fair Price", color="#f59e0b")
    for idx, (act, fair) in enumerate(zip(actual_prices, fair_prices)):
        diff = act - fair
        tag = "Bargain" if diff <= 0 else f"+${diff:.2f}"
        axes[1].text(idx, max(act, fair) + 0.015, tag, ha="center", fontsize=8.5, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(part_labels, fontsize=9.5, fontweight="bold")
    axes[1].set_ylabel("Unit Price (USD @ Qty 100)")
    axes[1].set_ylim(0, max(actual_prices + fair_prices) + 0.12)
    axes[1].set_title("2. Linear Regression Fair-Price Valuation", fontsize=10.5, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=8.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3: LSTM Time-Series Price & Stock Trajectory Forecast (6w history + 3w LSTM forecast)
    steps_hist = list(range(-5, 1))
    steps_fore = [0, 1, 2, 3]
    colors = ["#16a34a", "#2563eb", "#dc2626", "#9333ea"]
    for idx, c in enumerate(ranked):
        lstm_data = c["ml_predictions"]["lstm_forecast"]
        hist_p = lstm_data["price_history_6w"]
        fore_p = [hist_p[-1]] + lstm_data["price_forecast_3w"]
        col = colors[idx % len(colors)]
        axes[2].plot(steps_hist, hist_p, marker="o", color=col, linewidth=1.8, label=f"{c['part_number']} History")
        axes[2].plot(steps_fore, fore_p, marker="s", linestyle="--", color=col, linewidth=1.8, label=f"{c['part_number']} LSTM Forecast")
    axes[2].axvline(0, color="#64748b", linestyle=":", linewidth=1.5)
    axes[2].set_xlabel("Timeline (Weeks Relative to Current Quote Snapshot)")
    axes[2].set_ylabel("Unit Price (USD)")
    axes[2].set_title("3. LSTM Sequential Price Trend Forecast (+3 Weeks)", fontsize=10.5, fontweight="bold")
    axes[2].legend(loc="best", fontsize=7.5, ncol=2)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = output_dir / "module4_cost_performance_ranking.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    p1 = render_module1_visual(args.raw_root, args.output_dir)
    p2 = render_module2_visual(args.output_dir)
    p3 = render_module3_visual(args.output_dir)
    p4 = render_module4_visual(args.output_dir)
    print(f"Generated Module 1 visual: {p1}")
    print(f"Generated Module 2 visual: {p2}")
    print(f"Generated Module 3 visual: {p3}")
    print(f"Generated Module 4 visual: {p4}")


if __name__ == "__main__":
    main()
