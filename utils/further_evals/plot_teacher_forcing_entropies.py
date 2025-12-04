import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Add the parent directory to the path to import uncertainty_parser
sys.path.append(str(Path(__file__).parent.parent))
from uncertainty_parser import parse_uncertainty_metrics

method_to_results_dir = {
    "baseline": "compute_uncertainty_baseline",
    "Ours": {
        "k=1": "compute_uncertainty_random_choice_ablation_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_1_filtering_spatial_beam_search",
        "k=2": "compute_uncertainty_random_choice_ablation_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_2_filtering_spatial_beam_search",
        "k=3": "compute_uncertainty_random_choice_ablation_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_3_filtering_spatial_beam_search",
    },
    "random": {
        "k=1": "compute_uncertainty_svc_top_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_1_filtering_spatial_beam_search",
        "k=2": "compute_uncertainty_svc_top_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_2_filtering_spatial_beam_search",
        "k=3": "compute_uncertainty_svc_top_k_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_top_3_filtering_spatial_beam_search",
    },
    "MJ": {
        "k=1": "compute_uncertainty_svc_variable_claims_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_with_verifier_EQ_scorer_top_1_filtering_spatial_beam_search",
        "k=2": "compute_uncertainty_svc_variable_claims_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_with_verifier_EQ_scorer_top_2_filtering_spatial_beam_search",
        "k=3": "compute_uncertainty_svc_variable_claims_test_OpenGVLab_InternVL3-14B_50_1_8_8_2_with_verifier_EQ_scorer_top_3_filtering_spatial_beam_search",
    }
}

def parse_all_results(results_base_path="results"):
    """Parse uncertainty metrics for all methods and k values."""
    results = {}
    
    # Parse baseline results
    baseline_path = os.path.join(results_base_path, method_to_results_dir["baseline"])
    print(f"Parsing baseline results from: {baseline_path}")
    try:
        baseline_metrics = parse_uncertainty_metrics(baseline_path)
        results["baseline"] = baseline_metrics
    except Exception as e:
        print(f"Error parsing baseline: {e}")
        return None
    
    # Parse results for each method and k value
    for method_name, k_dict in method_to_results_dir.items():
        if method_name == "baseline":
            continue
            
        results[method_name] = {}
        for k, dir_name in k_dict.items():
            method_path = os.path.join(results_base_path, dir_name)
            print(f"Parsing {method_name} {k} results from: {method_path}")
            try:
                method_metrics = parse_uncertainty_metrics(method_path)
                results[method_name][k] = method_metrics
            except Exception as e:
                print(f"Error parsing {method_name} {k}: {e}")
                return None
    print(results)
    
    return results

def add_break_marks(ax, break_start, break_end, orientation='vertical'):
    """Add diagonal break marks to indicate axis break."""
    d = 0.015  # size of diagonal lines
    
    if orientation == 'vertical':
        # For y-axis breaks, draw on left side
        trans = ax.get_yaxis_transform()
        kwargs = dict(transform=trans, color='k', clip_on=False, linewidth=1.5)
        
        # Bottom break marks (at break_start in axis coordinates)
        y_pos = (break_start - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.plot((-d, +d), (y_pos - d, y_pos + d), **kwargs)
        ax.plot((1 - d, 1 + d), (y_pos - d, y_pos + d), **kwargs)

def create_entropy_plots(results, output_dir="plots", plot_mode="delta"):
    """Create three line plots for avg_answer_entropy across different categories with broken y-axes."""
    
    # Set up the plotting style for publication quality
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 21,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 16,
        'lines.linewidth': 3,
        'lines.markersize': 11,
        'axes.linewidth': 1.8,
        'grid.alpha': 0.3
    })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Categories to plot with their break intervals
    categories = ["overall", "correct", "wrong"]
    category_titles = ["overall", "correct answers", "wrong answers"]
    break_intervals = [
        (0.12, 0.18),  # overall
        (0.13, 0.179),  # correct
        (0.14, 0.187)   # wrong
    ]
    
    # Extract baseline values
    baseline_overall = results["baseline"]["overall"]["avg_answer_entropy"]
    baseline_correct = results["baseline"]["correct"]["avg_answer_entropy"]
    baseline_wrong = results["baseline"]["wrong"]["avg_answer_entropy"]
    baseline_values = [baseline_overall, baseline_correct, baseline_wrong]
    
    # Extract method values for different k values
    k_values = [1, 2, 3]
    methods = ["random", "MJ", "Ours"]
    method_colors = ["#2E86AB", "green", "#F18F01"]
    method_markers = ["o", "s", "X"]
    
    for i, (category, title, break_interval) in enumerate(zip(categories, category_titles, break_intervals)):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.08})
        
        # Set zorder for axes to control overlap
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        
        break_start, break_end = break_interval
        
        # Collect all y-values to determine appropriate ranges
        all_y_values = [baseline_values[i]]
        for method in methods:
            for k in k_values:
                k_key = f"k={k}"
                all_y_values.append(results[method][k_key][category]["avg_answer_entropy"])
        
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        
        # Set axis limits with breaks
        # Top subplot (upper range) - contains baseline
        # Add more padding above baseline to ensure dashed line is clearly visible
        padding_above = min(0.009, 0.08 * (y_max - break_end))
        ax1.set_ylim(break_end, baseline_values[i] + padding_above)
        # Bottom subplot (lower range) - contains method values
        ax2.set_ylim(y_min - 0.05 * (break_end - y_min), break_start)
        
        # Plot baseline on top subplot
        ax1.axhline(y=baseline_values[i], color='black', linestyle='--', 
                   linewidth=4, label='Baseline', alpha=0.8)
        
        # Plot methods on both subplots
        for method_idx, method in enumerate(methods):
            method_entropies = []
            for k in k_values:
                k_key = f"k={k}"
                method_entropies.append(results[method][k_key][category]["avg_answer_entropy"])
            
            # Plot on bottom subplot (where the data actually is)
            ax2.plot(k_values, method_entropies,
                    color=method_colors[method_idx],
                    marker=method_markers[method_idx],
                    linestyle='-',
                    linewidth=4,
                    markersize=13,
                    label=method,
                    markerfacecolor=method_colors[method_idx],
                    markeredgecolor='white',
                    markeredgewidth=1.5)
            
            # Also plot on top subplot (will be clipped, but ensures continuity if any points are there)
            ax1.plot(k_values, method_entropies,
                    color=method_colors[method_idx],
                    marker=method_markers[method_idx],
                    linestyle='-',
                    linewidth=4,
                    markersize=13,
                    markerfacecolor=method_colors[method_idx],
                    markeredgecolor='white',
                    markeredgewidth=1.5)
        
        # Customize the plots
        ax1.set_title(f'{title}', fontweight='bold', pad=20)
        ax2.set_xlabel('top-k (of world model\'s frames)', fontweight='bold')
        
        # Set y-label on the middle of the figure
        fig.text(0.04, 0.5, 'average answer entropy', va='center', rotation='vertical', 
                fontweight='bold', fontsize=20)
        
        # Set x-axis ticks and labels
        ax2.set_xticks(k_values)
        ax2.set_xticklabels([f'k={k}' for k in k_values])
        
        # Add grid
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Hide the spines between ax1 and ax2
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.tick_params(bottom=False)
        
        # Hide all y-axis labels on top subplot initially
        ax1.tick_params(left=False, labelleft=False)
        
        # Add a single y-axis label at the baseline value position
        ax1.set_yticks([baseline_values[i]])
        ax1.set_yticklabels([f'{baseline_values[i]:.3f}'])
        ax1.tick_params(left=True, labelleft=True)
        
        # Ensure bottom subplot has proper y-axis labels
        ax2.tick_params(left=True, labelleft=True)
        
        # Adjust y-axis label positioning to avoid overlap with break point
        ax2.tick_params(axis='y', pad=8)
        
        # Add break marks
        d = 0.015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.5)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        
        # Customize legend - combine handles from both subplots
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        
        # Combine and remove duplicates while preserving order
        all_handles = handles1 + handles2
        all_labels = labels1 + labels2
        
        # Remove duplicates (keep first occurrence)
        unique_labels = []
        unique_handles = []
        for handle, label in zip(all_handles, all_labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        if i == 0 or i == 1:
            legend = ax1.legend(unique_handles, unique_labels, loc='lower center', frameon=True, 
                            fancybox=True, shadow=False, framealpha=0.3, ncol=2)
        else:
            legend = ax1.legend(unique_handles, unique_labels, loc='lower center', frameon=True, 
                            fancybox=True, shadow=False, framealpha=0.3, ncol=2)
        legend.set_zorder(100)  # Ensure legend is on top

        # Add a horizontal line at the bottom of the legend
        legend_bbox = legend.get_window_extent()
        legend_bottom = legend_bbox.ymin / fig.get_figheight()
        ax2.axhline(y=legend_bottom, color='black', linewidth=2, clip_on=False)
        # Adjust layout
        plt.subplots_adjust(left=0.15)
        
        # Save the plot
        output_path = os.path.join(output_dir, f"avg_answer_entropy_{category}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved plot: {output_path}")
        
        # Also save as PDF for publication
        pdf_path = os.path.join(output_dir, f"avg_answer_entropy_{category}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved plot: {pdf_path}")
        
        plt.show()


def main():
    """Main function to run the analysis and create plots."""
    print("Starting teacher forcing entropy analysis...")
    
    # Parse all results
    results = parse_all_results()
    if results is None:
        print("Failed to parse results. Exiting.")
        return
    
    print("\nSuccessfully parsed all results!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for category in ["overall", "correct", "wrong"]:
        print(f"\n{category.upper()}:")
        print(f"  Baseline: {results['baseline'][category]['avg_answer_entropy']:.4f}")
        for method in ["random", "MJ", "Ours"]:
            print(f"  {method}:")
            for k in [1, 2, 3]:
                entropy = results[method][f"k={k}"][category]["avg_answer_entropy"]
                print(f"    k={k}: {entropy:.4f}")
    
    # Create plots
    print("\nCreating plots...")
    create_entropy_plots(results, plot_mode="")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()