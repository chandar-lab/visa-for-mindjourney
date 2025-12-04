import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Try to import uncertainty_parser from the parent dir; be robust to __file__ absence
try:
    THIS_DIR = Path(__file__).resolve().parent
    sys.path.append(str(THIS_DIR.parent))
except NameError:
    # __file__ may be undefined in some environments (e.g., notebooks)
    sys.path.append(str(Path.cwd().parent))

from uncertainty_parser import parse_uncertainty_metrics

# Map method -> subdir(s)
method_to_results_dir = {
    "baseline": "compute_uncertainty_baseline",
    "Ours": {
        "k=1": "compute_uncertainty_random_choice_ablation_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_1_filtering_spatial_beam_search",
        "k=2": "compute_uncertainty_random_choice_ablation_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_2_filtering_spatial_beam_search",
        "k=3": "compute_uncertainty_random_choice_ablation_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_3_filtering_spatial_beam_search",
    },
    "MJ": {
        "k=1": "compute_uncertainty_svc_top_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_1_filtering_spatial_beam_search",
        "k=2": "compute_uncertainty_svc_top_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_2_filtering_spatial_beam_search",
        "k=3": "compute_uncertainty_svc_top_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_3_filtering_spatial_beam_search",
    },
    "random": {
        "k=1": "compute_uncertainty_svc_variable_claims_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_with_verifier_EQ_scorer_top_1_filtering_spatial_beam_search",
        "k=2": "compute_uncertainty_svc_variable_claims_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_with_verifier_EQ_scorer_top_2_filtering_spatial_beam_search",
        "k=3": "compute_uncertainty_svc_variable_claims_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_with_verifier_EQ_scorer_top_3_filtering_spatial_beam_search",
    }
}

def parse_all_results(results_base_path="results"):
    """Parse uncertainty metrics for all methods and k values."""
    results = {}

    # Baseline (single dir)
    baseline_path = os.path.join(results_base_path, method_to_results_dir["baseline"])
    print(f"Parsing baseline from: {baseline_path}")
    baseline_metrics = parse_uncertainty_metrics(baseline_path)
    results["baseline"] = baseline_metrics

    # Others
    for method_name, k_dict in method_to_results_dir.items():
        if method_name == "baseline":
            continue
        results[method_name] = {}
        for k, dir_name in k_dict.items():
            method_path = os.path.join(results_base_path, dir_name)
            print(f"Parsing {method_name} {k} from: {method_path}")
            results[method_name][k] = parse_uncertainty_metrics(method_path)

    return results

def create_entropy_line_plots(results, output_dir="plots", plot_mode="absolute"):
    """
    Create three line plots (overall, correct, wrong) of avg_answer_entropy
    vs top-k for methods: random, MJ, Ours. Baseline shown as a dashed line.

    plot_mode:
      - "absolute": plot raw entropies; baseline is a dashed horizontal line.
      - "delta":    plot (method - baseline); baseline is 0.
    """
    plot_mode = plot_mode.lower().strip() or "absolute"
    if plot_mode not in {"absolute", "delta"}:
        print(f"[WARN] Unrecognized plot_mode='{plot_mode}', defaulting to 'absolute'.")
        plot_mode = "absolute"

    os.makedirs(output_dir, exist_ok=True)

    categories = ["overall", "correct", "wrong"]
    titles = ["overall", "correct answers", "wrong answers"]
    methods = ["random", "MJ", "Ours"]
    markers = ["o", "s", "^"]
    # Keep colors aligned with your figure: blue, purple, orange
    colors = ["#2E86AB", "green", "#F18F01"]

    # Baseline values (same across k)
    baseline_vals = {
        cat: results["baseline"][cat]["avg_answer_entropy"] for cat in categories
    }

    k_list = [1, 2, 3]
    x = np.array(k_list, dtype=float)

    # Matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 21,
        'axes.titlesize': 22,
        'axes.labelsize': 21,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18,
        'figure.titlesize': 20,
        'lines.linewidth': 3,
        'lines.markersize': 9,
        'axes.linewidth': 1.8,
        'grid.alpha': 0.3
    })

    for cat, title in zip(categories, titles):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Baseline (dashed line)
        if plot_mode == "absolute":
            y_base = baseline_vals[cat]
            ax.axhline(y=y_base, linestyle="--", linewidth=3, color="black", label="Baseline")
        else:
            y_base = 0.0
            ax.axhline(y=0.0, linestyle="--", linewidth=3, color="black", label="Baseline (0)")

        # Plot each method as a line with markers
        for m, marker, color in zip(methods, markers, colors):
            y = []
            for k in k_list:
                val = results[m][f"k={k}"][cat]["avg_answer_entropy"]
                if plot_mode == "delta":
                    val = val - baseline_vals[cat]
                y.append(val)
            ax.plot(x, y, marker=marker, label=m, color=color)

        ax.set_xlabel("top-k (of world model's frames)")
        ax.set_ylabel("average answer entropy" if plot_mode == "absolute"
                      else "Δ average answer entropy (vs baseline)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in k_list])

        ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9, loc="upper left")
        fig.tight_layout()

        suffix = "" if plot_mode == "absolute" else "_delta"
        png_path = os.path.join(output_dir, f"avg_answer_entropy_{cat}{suffix}.png")
        pdf_path = os.path.join(output_dir, f"avg_answer_entropy_{cat}{suffix}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

        plt.close(fig)

def main():
    print("Starting teacher-forcing entropy analysis...")
    results = parse_all_results()
    print("Parsed all results successfully.\n")

    # Summary stats
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for cat in ["overall", "correct", "wrong"]:
        print(f"\n{cat.upper()}:")
        print(f"  Baseline: {results['baseline'][cat]['avg_answer_entropy']:.4f}")
        for method in ["random", "MJ", "Ours"]:
            print(f"  {method}:")
            for k in [1, 2, 3]:
                v = results[method][f'k={k}'][cat]['avg_answer_entropy']
                print(f"    k={k}: {v:.4f}")

    print("\nCreating plots...")
    # Choose "absolute" to match your shared figure; use "delta" if you prefer deltas vs baseline
    create_entropy_line_plots(results, plot_mode="absolute")
    print("Done.")

if __name__ == "__main__":
    main()
