#!/usr/bin/env python3
"""
Action Distribution Parsing and Plotting Script

This script parses chosen actions from MindJourney and Verifier results and creates
visualizations comparing action distributions across different top-k values.

Usage:
    python parse_and_plot_action_dist.py --results_dir results/ --output_dir plots/
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import argparse

# Try to import matplotlib and numpy, but make them optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib and/or numpy not available. Plotting will be skipped.")


def parse_chosen_actions(gpt_file_path: str) -> List[str]:
    """
    Parse the chosen helpful actions from a gpt.json file.
    The chosen actions are embedded in the prompt content.
    
    Args:
        gpt_file_path: Path to the gpt.json file
        
    Returns:
        List of chosen action strings
    """
    with open(gpt_file_path, 'r') as f:
        data = json.load(f)
    
    chosen_actions = []
    prompt_content = data.get('prompt', {}).get('content', [])
    
    # Look for actions in the prompt content
    for content_item in prompt_content:
        if isinstance(content_item, list) and len(content_item) >= 2:
            # Check if this looks like an action line (contains action text and image path)
            action_text = content_item[0]
            if isinstance(action_text, str):
                # Look for various action patterns
                if ("meters" in action_text or "degrees" in action_text or 
                    "turn" in action_text or "move" in action_text):
                    # Extract the action (e.g., "move forward 0.25 meters", "turn right 9 degrees")
                    chosen_actions.append(action_text.strip())
    
    return chosen_actions


def extract_action_type(action_string: str) -> str:
    """
    Extract the action type from an action string.
    
    Args:
        action_string: Action string like "move forward 0.25 meters" or "turn left 9 degrees"
        
    Returns:
        Action type: "move forward", "turn left", or "turn right"
    """
    action_string = action_string.lower()
    
    if "move forward" in action_string:
        return "move forward"
    elif "turn left" in action_string:
        return "turn left"
    elif "turn right" in action_string:
        return "turn right"
    else:
        # Fallback: try to extract from the beginning of the string
        if "move" in action_string:
            return "move forward"
        elif "left" in action_string:
            return "turn left"
        elif "right" in action_string:
            return "turn right"
        else:
            return "unknown"


def extract_action_magnitude(action_string: str) -> float:
    """
    Extract the magnitude from an action string.
    
    Args:
        action_string: Action string like "move forward 0.25 meters" or "turn left 9 degrees"
        
    Returns:
        Magnitude value as float
    """
    import re
    
    # Look for numbers in the action string
    numbers = re.findall(r'\d+(?:\.\d+)?', action_string)
    if numbers:
        return float(numbers[0])
    return 0.0


def get_magnitude_bucket(magnitude: float, action_type: str) -> str:
    """
    Categorize magnitude into buckets based on action type.
    
    Args:
        magnitude: The magnitude value
        action_type: The action type ("move forward", "turn left", "turn right")
        
    Returns:
        Bucket name for the magnitude
    """
    if action_type == "move forward":
        if magnitude <= 0.25:
            return "0.25m"
        elif magnitude <= 0.5:
            return "0.5m"
        else:
            return "0.75m"
    else:  # turn left or turn right
        if magnitude <= 9:
            return "9°"
        elif magnitude <= 18:
            return "18°"
        else:
            return "27°"


def parse_results_directory(results_dir: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Parse all results directories and extract action distributions.
    
    Args:
        results_dir: Path to the results directory containing all model results
        
    Returns:
        Dictionary with structure: data[model][top_k][action_type] = count
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")
    
    # Initialize data structure
    data = {
        "MindJourney": {
            "top_1": defaultdict(int),
            "top_2": defaultdict(int), 
            "top_3": defaultdict(int),
            "top_4": defaultdict(int)
        },
        "Ours": {
            "top_1": defaultdict(int),
            "top_2": defaultdict(int),
            "top_3": defaultdict(int), 
            "top_4": defaultdict(int)
        }
    }
    
    # Find all result directories
    result_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    for result_dir in result_dirs:
        dir_name = result_dir.name
        
        # Determine model type and top-k value
        if dir_name.startswith("svc_test_OpenGVLab_InternVL3-14B_150"):
            model = "MindJourney"
            # Extract top-k from directory name
            if "top_1" in dir_name:
                top_k = "top_1"
            elif "top_2" in dir_name:
                top_k = "top_2"
            elif "top_3" in dir_name:
                top_k = "top_3"
            elif "top_4" in dir_name:
                top_k = "top_4"
            else:
                print(f"Warning: Could not determine top-k for {dir_name}")
                continue
                
        elif dir_name.startswith("svc_variable_claims_test_OpenGVLab"):
            model = "Ours"
            # Extract top-k from directory name
            if "top_1" in dir_name:
                top_k = "top_1"
            elif "top_2" in dir_name:
                top_k = "top_2"
            elif "top_3" in dir_name:
                top_k = "top_3"
            elif "top_4" in dir_name:
                top_k = "top_4"
            else:
                print(f"Warning: Could not determine top-k for {dir_name}")
                continue
        else:
            print(f"Warning: Unknown directory pattern: {dir_name}")
            continue
        
        print(f"Processing {model} {top_k}: {dir_name}")
        
        # Process all question directories
        question_dirs = [d for d in result_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        
        for question_dir in question_dirs:
            gpt_file = question_dir / "gpt.json"
            if gpt_file.exists():
                try:
                    chosen_actions = parse_chosen_actions(str(gpt_file))
                    
                    # Count action types
                    for action in chosen_actions:
                        action_type = extract_action_type(action)
                        if action_type != "unknown":
                            data[model][top_k][action_type] += 1
                            
                except Exception as e:
                    print(f"Error processing {gpt_file}: {e}")
                    continue
    
    # Convert defaultdicts to regular dicts
    for model in data:
        for top_k in data[model]:
            data[model][top_k] = dict(data[model][top_k])
    
    return data


def parse_results_directory_fine_grained(results_dir: str) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
    """
    Parse all results directories and extract fine-grained action distributions.
    
    Args:
        results_dir: Path to the results directory containing all model results
        
    Returns:
        Dictionary with structure: data[model][top_k][action_type][magnitude_bucket] = count
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")
    
    # Initialize data structure
    data = {
        "MindJourney": {
            "top_1": defaultdict(lambda: defaultdict(int)),
            "top_2": defaultdict(lambda: defaultdict(int)), 
            "top_3": defaultdict(lambda: defaultdict(int)),
            "top_4": defaultdict(lambda: defaultdict(int))
        },
        "Ours": {
            "top_1": defaultdict(lambda: defaultdict(int)),
            "top_2": defaultdict(lambda: defaultdict(int)),
            "top_3": defaultdict(lambda: defaultdict(int)), 
            "top_4": defaultdict(lambda: defaultdict(int))
        }
    }
    
    # Find all result directories
    result_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    for result_dir in result_dirs:
        dir_name = result_dir.name
        
        # Determine model type and top-k value
        if dir_name.startswith("svc_test_OpenGVLab_InternVL3-14B_150"):
            model = "MindJourney"
            # Extract top-k from directory name
            if "top_1" in dir_name:
                top_k = "top_1"
            elif "top_2" in dir_name:
                top_k = "top_2"
            elif "top_3" in dir_name:
                top_k = "top_3"
            elif "top_4" in dir_name:
                top_k = "top_4"
            else:
                print(f"Warning: Could not determine top-k for {dir_name}")
                continue
                
        elif dir_name.startswith("svc_variable_claims_test_OpenGVLab"):
            model = "Ours"
            # Extract top-k from directory name
            if "top_1" in dir_name:
                top_k = "top_1"
            elif "top_2" in dir_name:
                top_k = "top_2"
            elif "top_3" in dir_name:
                top_k = "top_3"
            elif "top_4" in dir_name:
                top_k = "top_4"
            else:
                print(f"Warning: Could not determine top-k for {dir_name}")
                continue
        else:
            print(f"Warning: Unknown directory pattern: {dir_name}")
            continue
        
        print(f"Processing {model} {top_k}: {dir_name}")
        
        # Process all question directories
        question_dirs = [d for d in result_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        
        for question_dir in question_dirs:
            gpt_file = question_dir / "gpt.json"
            if gpt_file.exists():
                try:
                    chosen_actions = parse_chosen_actions(str(gpt_file))
                    
                    # Count action types with magnitude buckets
                    for action in chosen_actions:
                        action_type = extract_action_type(action)
                        magnitude = extract_action_magnitude(action)
                        magnitude_bucket = get_magnitude_bucket(magnitude, action_type)
                        
                        if action_type != "unknown":
                            data[model][top_k][action_type][magnitude_bucket] += 1
                            
                except Exception as e:
                    print(f"Error processing {gpt_file}: {e}")
                    continue
    
    # Convert defaultdicts to regular dicts
    for model in data:
        for top_k in data[model]:
            for action_type in data[model][top_k]:
                data[model][top_k][action_type] = dict(data[model][top_k][action_type])
    
    return data


def create_fine_grained_action_distribution_plot(data: Dict[str, Dict[str, Dict[str, Dict[str, int]]]], 
                                                output_dir: str = "plots") -> None:
    """
    Create fine-grained action distribution plot with 3x3 subplots for each model/top-k combination.
    
    Args:
        data: Dictionary with structure data[model][top_k][action_type][magnitude_bucket] = count
        output_dir: Directory to save plots
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting not available. Skipping visualization.")
        return
    
    # Set up the plotting style for publication quality
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 19,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 19,
        'legend.fontsize': 19,
        'figure.titlesize': 22,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.linewidth': 1.5,
        'grid.alpha': 0.2
    })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define data structure
    models = ["MindJourney", "Ours"]
    top_k_values = ["top_1", "top_2", "top_3", "top_4"]
    action_types = ["move forward", "turn left", "turn right"]
    magnitude_buckets = {
        "move forward": ["0.25m", "0.5m", "0.75m"],
        "turn left": ["9°", "18°", "27°"],
        "turn right": ["9°", "18°", "27°"]
    }
    
    # Create custom colormap - vibrant gradient from light to dark
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#1565C0']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_blues', colors_list, N=n_bins)
    
    # Create the main plot with more spacing
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 4, hspace=0.2, wspace=0.5, 
                          left=0.08, right=0.85, top=0.90, bottom=0.12)
    
    # Track max value for consistent colorbar
    max_value = 0
    all_heatmaps = []
    
    for i, model in enumerate(models):
        for j, top_k in enumerate(top_k_values):
            ax = fig.add_subplot(gs[i, j])
            
            # Create 3x3 grid within this subplot
            heatmap_data = np.zeros((3, 3))
            
            for row, action_type in enumerate(action_types):
                for col, magnitude_bucket in enumerate(magnitude_buckets[action_type]):
                    count = data[model][top_k][action_type].get(magnitude_bucket, 0)
                    heatmap_data[row, col] = count
            
            max_value = max(max_value, heatmap_data.max())
            
            # Create heatmap with enhanced styling
            im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', 
                          interpolation='nearest', vmin=0, vmax=None)
            all_heatmaps.append(im)
            
            # Add subtle grid lines
            for spine in ax.spines.values():
                spine.set_edgecolor('#CCCCCC')
                spine.set_linewidth(2)
            
            # Set ticks and labels with better styling
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels([])
            ax.set_yticklabels(action_types, fontsize=15, fontweight='500')
            
            # Add cell borders for clarity
            for row in range(3):
                for col in range(3):
                    rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, 
                                        fill=False, edgecolor='white', 
                                        linewidth=2, alpha=0.6)
                    ax.add_patch(rect)
            
            # Add text annotations with improved visibility
            for row in range(3):
                for col in range(3):
                    value = int(heatmap_data[row, col])
                    if value > 0:
                        # Determine text color based on background intensity
                        text_color = 'white' if value > heatmap_data.max()*0.5 else '#1A1A1A'
                        ax.text(col, row, str(value), ha='center', va='center', 
                               fontweight='bold', fontsize=19, color=text_color,
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='none', 
                                       edgecolor='none'))
            
            # Enhanced subplot title with model and top-k info
            title_text = f"{model}\n{top_k.replace('_', '-')}"
            ax.set_title(title_text, fontweight='bold', fontsize=16, 
                        pad=12, color='#2C3E50')
            
            # Remove ticks for cleaner look
            ax.tick_params(axis='both', which='both', length=0)
    
    # Normalize all heatmaps to same scale
    for im in all_heatmaps:
        im.set_clim(0, max_value)
    
    # Add overall title with better positioning
    fig.suptitle("Action magnitude distribution: MindJourney vs Ours", 
                 fontsize=22, fontweight='bold', y=0.98, color='#1A1A1A')
    
    # Add enhanced colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(all_heatmaps[0], cax=cbar_ax)
    cbar.set_label('action count', fontweight='bold', fontsize=19, 
                   labelpad=15, color='#2C3E50')
    cbar.ax.tick_params(labelsize=14, colors='#2C3E50')
    
    # Style the colorbar
    cbar.outline.set_linewidth(1.5)
    cbar.outline.set_edgecolor('#CCCCCC')
    
    # Add arrow with label at the bottom using figure coordinates
    fig.text(0.36, 0.04, 'increasing order of action magnitudes', ha='center', va='center', 
             fontweight='bold', fontsize=20, color='#C0392B',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', 
                      edgecolor='#C0392B', linewidth=1.5, alpha=0.8))
    
    # Add fancy arrow using FancyArrowPatch
    from matplotlib.patches import FancyArrowPatch
    arrow = FancyArrowPatch((0.31, 0.09), (0.43, 0.09),
                           transform=fig.transFigure,
                           arrowstyle='->,head_width=0.6,head_length=0.8', 
                           mutation_scale=25,
                           linewidth=3.5,
                           color='#C0392B',
                           alpha=0.9,
                           zorder=1000)
    fig.add_artist(arrow)
    
    # Add subtle shadow effect to arrow
    arrow_shadow = FancyArrowPatch((0.311, 0.089), (0.431, 0.089),
                                  transform=fig.transFigure,
                                  arrowstyle='->,head_width=0.6,head_length=0.8', 
                                  mutation_scale=25,
                                  linewidth=4,
                                  color='black',
                                  alpha=0.2,
                                  zorder=999)
    fig.add_artist(arrow_shadow)
    
    # Save the plot
    output_path = os.path.join(output_dir, "fine_grained_action_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f"Saved plot: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(output_dir, "fine_grained_action_distribution.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f"Saved plot: {pdf_path}")
    
    plt.show()


def create_action_distribution_plot(data: Dict[str, Dict[str, Dict[str, int]]], 
                                  output_dir: str = "plots") -> None:
    """
    Create grouped bar chart comparing action distributions.
    
    Args:
        data: Dictionary with structure data[model][top_k][action_type] = count
        output_dir: Directory to save plots
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting not available. Skipping visualization.")
        return
    
    # Set up the plotting style for publication quality
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 19,
        'ytick.labelsize': 19,
        'legend.fontsize': 19,
        'figure.titlesize': 22,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.25
    })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define data structure
    models = ["MindJourney", "Ours"]
    top_k_values = ["top_1", "top_2", "top_3", "top_4"]
    action_types = ["move forward", "turn left", "turn right"]
    
    # Enhanced color palette with gradients
    colors = ["#3498DB", "#E74C3C", "#F39C12"]  # Vibrant Blue, Red, Orange
    edge_colors = ["#2980B9", "#C0392B", "#E67E22"]  # Darker shades for edges
    
    # Create the plot with better spacing
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.40, wspace=0.30, 
                          left=0.08, right=0.95, top=0.85, bottom=0.15)
    
    for i, model in enumerate(models):
        for j, top_k in enumerate(top_k_values):
            ax = fig.add_subplot(gs[i, j])
            
            # Extract data for this model/top_k combination
            counts = [data[model][top_k].get(action_type, 0) for action_type in action_types]
            
            # Create x positions for bars
            x_pos = np.arange(len(action_types))
            
            # Create grouped bars with enhanced styling
            bars = ax.bar(x_pos, counts, color=colors, alpha=0.85, 
                         edgecolor=edge_colors, linewidth=2.5, width=0.7)
            
            # Add gradient effect to bars
            for bar, color, edge_color in zip(bars, colors, edge_colors):
                bar.set_linewidth(2.5)
                bar.set_edgecolor(edge_color)
                # Add subtle shadow
                ax.bar(bar.get_x(), bar.get_height(), bar.get_width(), 
                      bottom=0, color='black', alpha=0.1, zorder=0)
            
            # Customize subplot with enhanced styling
            title_text = f"{model}\n{top_k.replace('_', '-')}"
            ax.set_title(title_text, fontweight='bold', fontsize=15, 
                        pad=15, color='#2C3E50')
            
            ax.set_ylabel("action count", fontweight='bold', fontsize=14, 
                         color='#2C3E50', labelpad=10)
            
            # Set x-axis
            ax.set_xticks(x_pos)
            ax.set_xticklabels([])  # Remove x-axis tick labels
            
            # Add value labels on bars with enhanced styling
            for idx, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                if height > 0:  # Only add labels for non-zero counts
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{count}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=13, color='#2C3E50',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   edgecolor=edge_colors[idx],
                                   linewidth=1.5,
                                   alpha=0.9))
            
            # Set y-axis limits with some padding
            max_count = max(counts) if counts else 1
            ax.set_ylim(0, max_count * 1.15)
            
            # Enhanced grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, 
                   color='#95A5A6', axis='y')
            ax.set_axisbelow(True)
            
            # Style spines
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('#BDC3C7')
                ax.spines[spine].set_linewidth(1.5)
            
            # Style tick parameters
            ax.tick_params(axis='both', which='major', labelsize=13, 
                          colors='#2C3E50', length=0)
    
    # Add overall title with better positioning
    fig.suptitle("Action Distribution Comparison: MindJourney vs Ours", 
                 fontsize=22, fontweight='bold', y=0.98, color='#1A1A1A')
    
    # Create enhanced legend with fancy styling
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[i], edgecolor=edge_colors[i], 
              linewidth=2.5, alpha=0.85, label=action_types[i])
        for i in range(3)
    ]
    
    legend = fig.legend(handles=legend_elements, 
                       loc='upper center',
                       bbox_to_anchor=(0.5, 0.08), 
                       ncol=3, 
                       fontsize=18,
                       frameon=True,
                       fancybox=True,
                       shadow=True,
                       framealpha=0.95,
                       edgecolor='#BDC3C7',
                       borderpad=1,
                       columnspacing=2,
                       handlelength=2,
                       handleheight=1.5)
    
    # Style legend frame
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_facecolor('#F8F9FA')
    
    # Save the plot
    output_path = os.path.join(output_dir, "action_distribution_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f"Saved plot: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(output_dir, "action_distribution_comparison.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print(f"Saved plot: {pdf_path}")
    
    plt.show()


def print_summary_statistics(data: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    """
    Print summary statistics of the action distributions.
    
    Args:
        data: Dictionary with action distribution data
    """
    print("\n" + "="*80)
    print("ACTION DISTRIBUTION SUMMARY")
    print("="*80)
    
    for model in ["MindJourney", "Ours"]:
        print(f"\n{model}:")
        print("-" * 40)
        
        for top_k in ["top_1", "top_2", "top_3", "top_4"]:
            print(f"  {top_k.replace('_', '-')}:")
            total_actions = sum(data[model][top_k].values())
            
            for action_type in ["move forward", "turn left", "turn right"]:
                count = data[model][top_k].get(action_type, 0)
                percentage = (count / total_actions * 100) if total_actions > 0 else 0
                print(f"    {action_type}: {count} ({percentage:.1f}%)")
            
            print(f"    Total: {total_actions}")


def save_data_to_json(data: Dict[str, Dict[str, Dict[str, int]]], output_path: str) -> None:
    """
    Save the parsed data to a JSON file for further analysis.
    
    Args:
        data: Dictionary with action distribution data
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to: {output_path}")


def main():
    """Main function to run the action distribution analysis."""
    parser = argparse.ArgumentParser(description="Parse and plot action distributions from MindJourney results")
    parser.add_argument("--results_dir", type=str, default="results/",
                       help="Path to results directory containing all model results")
    parser.add_argument("--output_dir", type=str, default="plots/",
                       help="Directory to save plots")
    parser.add_argument("--save_data", type=str, default=None,
                       help="Path to save parsed data as JSON (optional)")
    parser.add_argument("--fine_grained", action="store_true",
                       help="Generate fine-grained analysis with magnitude buckets")
    
    args = parser.parse_args()
    
    print("Starting action distribution analysis...")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Fine-grained analysis: {args.fine_grained}")
    
    try:
        # Parse all results
        data = parse_results_directory(args.results_dir)
        
        # Print summary statistics
        print_summary_statistics(data)
        
        # Create basic visualization
        create_action_distribution_plot(data, args.output_dir)
        
        # Create fine-grained visualization if requested
        if args.fine_grained:
            print("\nGenerating fine-grained analysis...")
            fine_grained_data = parse_results_directory_fine_grained(args.results_dir)
            create_fine_grained_action_distribution_plot(fine_grained_data, args.output_dir)
        
        # Save data if requested
        if args.save_data:
            save_data_to_json(data, args.save_data)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()