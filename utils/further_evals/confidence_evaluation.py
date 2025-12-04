#!/usr/bin/env python3
"""
Confidence Evaluation Script for Verifier Results

This script evaluates the confidence of the verifier's results by:
1. Parsing chosen helpful actions for each question from gpt.json files
2. Extracting confidence metrics (average_confidence and evidence_quality_score) for selected actions
3. Computing final average confidence and evidence quality scores for each question
4. Determining correctness labels from gpt.json results
5. Computing calibration metrics (ECE, Brier score, NLL) for both confidence metrics
"""

import json
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse

# Try to import matplotlib and numpy, but make them optional
try:
    import matplotlib.pyplot as plt
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


def extract_confidence_metrics(results_data: Dict, q_id: str, chosen_actions: List[str]) -> Tuple[List[float], List[float]]:
    """
    Extract confidence metrics for the chosen actions from results.json data.
    
    Args:
        results_data: The loaded results.json data
        q_id: Question ID
        chosen_actions: List of chosen action strings
        
    Returns:
        Tuple of (average_confidences, evidence_quality_scores) for chosen actions
    """
    confidences = []
    evidence_scores = []
    
    verification_metrics = results_data.get('verification_metrics', {})
    q_data = verification_metrics.get(q_id, {})
    
    parsed_actions_count = 0
    # Iterate through step IDs (usually just "0")
    for step_id, step_data in q_data.items():
        # Iterate through high-level actions (e.g., "move forward")
        for high_level_action, action_data in step_data.items():
            if high_level_action in ['helpful_score_threshold', 'exploration_score_threshold']:
                continue
                
            # Iterate through exact actions (e.g., "move forward 0.25 meters")
            for exact_action, metrics in action_data.items():
                if isinstance(metrics, dict):
                    # Check if this exact action was chosen
                    if exact_action in chosen_actions:
                        avg_conf = metrics.get('average_confidence', 0.0)
                        evidence_score = metrics.get('evidence_quality_score', 0.0)
                        confidences.append(avg_conf)
                        evidence_scores.append(evidence_score)
                        parsed_actions_count += 1
    
    assert parsed_actions_count == len(chosen_actions), f"Parsed {parsed_actions_count} actions for question {q_id}, but expected {len(chosen_actions)}"
    return confidences, evidence_scores


def get_correctness_label(gpt_file_path: str) -> int:
    """
    Get the correctness label (1 for correct, 0 for wrong) from gpt.json.
    
    Args:
        gpt_file_path: Path to the gpt.json file
        
    Returns:
        1 if correct, 0 if wrong
    """
    with open(gpt_file_path, 'r') as f:
        data = json.load(f)
    
    result = data.get('result', '')
    return 1 if result == 'correct' else 0


def compute_calibration_metrics(confidences: List[float], labels: List[int]) -> Tuple[float, float, float]:
    """
    Compute Expected Calibration Error (ECE), Brier Score, and Negative Log-Likelihood.
    
    Args:
        confidences: List of confidence scores
        labels: List of binary labels (0 or 1)
        
    Returns:
        Tuple of (ECE, Brier Score, NLL)
    """
    n = len(confidences)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # ECE computation
    n_bins = 10
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = []
        for j in range(n):
            if bin_lower < confidences[j] <= bin_upper:
                in_bin.append(j)
        
        if len(in_bin) > 0:
            prop_in_bin = len(in_bin) / n
            accuracy_in_bin = sum(labels[j] for j in in_bin) / len(in_bin)
            avg_confidence_in_bin = sum(confidences[j] for j in in_bin) / len(in_bin)
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    # Brier Score
    brier_score = sum((confidences[i] - labels[i]) ** 2 for i in range(n)) / n
    
    # Negative Log-Likelihood
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    nll = 0.0
    for i in range(n):
        conf_clipped = max(epsilon, min(1 - epsilon, confidences[i]))
        nll += -(labels[i] * math.log(conf_clipped) + (1 - labels[i]) * math.log(1 - conf_clipped))
    nll /= n
    
    return ece, brier_score, nll


def create_calibration_plot(confidences: List[float], labels: List[int], 
                          title: str, save_path: str, n_bins: int = 10):
    """
    Create an accuracy vs confidence plot using equal-mass bins.
    
    Args:
        confidences: List of confidence scores
        labels: List of binary labels (0 or 1)
        title: Title for the plot
        save_path: Path to save the plot
        n_bins: Number of bins to use
    """
    if not PLOTTING_AVAILABLE:
        print(f"Skipping plot creation for {title} - matplotlib not available")
        return
        
    confidences = np.array(confidences)
    labels = np.array(labels)
    
    # Check if we have sufficient variance for equal-mass binning
    conf_std = np.std(confidences)
    conf_range = np.max(confidences) - np.min(confidences)
    
    print(f"Confidence statistics for {title}:")
    print(f"  Mean: {np.mean(confidences):.6f}")
    print(f"  Std: {conf_std:.6f}")
    print(f"  Range: {conf_range:.6f}")
    print(f"  Min: {np.min(confidences):.6f}, Max: {np.max(confidences):.6f}")
    
    # If variance is very low, use equal-width bins instead
    if conf_std < 0.01 or conf_range < 0.01:
        print(f"Warning: Low confidence variance. Using equal-width bins.")
        # Create equal-width bins
        bin_boundaries = np.linspace(np.min(confidences) - 0.001, np.max(confidences) + 0.001, n_bins + 1)
    else:
        # Create equal-mass bins
        bin_boundaries = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
        bin_boundaries[0] = -np.inf  # Ensure all values are included
        bin_boundaries[-1] = np.inf
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            bin_accuracy = np.mean(labels[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
            
            print(f"  Bin {i}: [{bin_boundaries[i]:.6f}, {bin_boundaries[i+1]:.6f}) -> n={bin_count}, conf={bin_confidence:.6f}, acc={bin_accuracy:.3f}")
        else:
            print(f"  Bin {i}: [{bin_boundaries[i]:.6f}, {bin_boundaries[i+1]:.6f}) -> n=0 (empty)")
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot accuracy vs confidence
    plt.subplot(2, 1, 1)
    
    # Use scatter plot instead of line plot for better visualization
    plt.scatter(bin_confidences, bin_accuracies, s=[count*10 for count in bin_counts], 
               c='blue', alpha=0.7, label='Binned Accuracy (size=count)')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy vs Confidence\n(Point size represents sample count)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.9, 1.0)  # Zoom in on the actual range
    plt.ylim(0, 1)
    
    # Add bin counts as text annotations
    for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
        plt.annotate(f'n={count}', (conf, acc), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10, alpha=0.8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Plot bin counts
    plt.subplot(2, 1, 2)
    plt.bar(range(len(bin_counts)), bin_counts, alpha=0.7, color='skyblue')
    plt.xlabel('Bin Index')
    plt.ylabel('Number of Samples')
    plt.title('Sample Count per Bin')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Calibration plot saved to: {save_path}")
    
    # Print interpretation summary
    if len(bin_counts) == 2 and sum(1 for c in bin_counts if c > 0) == 2:
        print(f"  Interpretation: The verifier shows poor calibration with only 2 distinct confidence levels:")
        print(f"    - {bin_counts[0]} samples with confidence {bin_confidences[0]:.3f} → accuracy {bin_accuracies[0]:.3f}")
        print(f"    - {bin_counts[1]} samples with confidence {bin_confidences[1]:.3f} → accuracy {bin_accuracies[1]:.3f}")
        print(f"    - The verifier is overconfident: higher confidence doesn't correlate with higher accuracy")


def main():
    parser = argparse.ArgumentParser(description='Evaluate confidence of verifier results')
    parser.add_argument('--results_dir', type=str, 
                       default='results/svc_variable_claims_test_OpenGVLab_InternVL3-14B_150_1_8_8_2_with_verifier_EQ_scorer_top_1_filtering_spatial_beam_search',
                       help='Path to the results directory')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_json_path = results_dir / 'results.json'
    
    # Load the main results.json file
    print(f"Loading results from {results_json_path}")
    with open(results_json_path, 'r') as f:
        results_data = json.load(f)
    
    # Get all question IDs from the verification metrics
    verification_metrics = results_data.get('verification_metrics', {})
    question_ids = list(verification_metrics.keys())
    print(f"Found {len(question_ids)} questions")
    
    # Collect data for all questions
    all_avg_confidences = []
    all_evidence_scores = []
    all_labels = []
    
    for q_id in question_ids:
        gpt_file_path = results_dir / q_id / 'gpt.json'
        
        if not gpt_file_path.exists():
            print(f"Warning: gpt.json not found for question {q_id}, skipping")
            continue
        
        # Parse chosen actions from gpt.json
        chosen_actions = parse_chosen_actions(str(gpt_file_path))
        
        if not chosen_actions:
            print(f"Warning: No chosen actions found for question {q_id}, skipping")
            continue
        
        # Extract confidence metrics for chosen actions
        confidences, evidence_scores = extract_confidence_metrics(results_data, q_id, chosen_actions)
        
        if not confidences:
            print(f"Warning: No confidence metrics found for question {q_id}, skipping")
            continue
        
        # Compute final average confidence and evidence quality score
        final_avg_confidence = sum(confidences) / len(confidences)
        final_evidence_score = sum(evidence_scores) / len(evidence_scores)
        
        # Get correctness label
        label = get_correctness_label(str(gpt_file_path))
        
        all_avg_confidences.append(final_avg_confidence)
        all_evidence_scores.append(final_evidence_score)
        all_labels.append(label)
        
        print(f"Question {q_id}: avg_conf={final_avg_confidence:.4f}, evidence_score={final_evidence_score:.4f}, correct={label}")
    
    print(f"\nProcessed {len(all_avg_confidences)} questions")
    
    if len(all_avg_confidences) == 0:
        print("No questions were processed. Please check the data format and thresholds.")
        return
    
    # Compute calibration metrics for average confidence
    print("\n=== Calibration Metrics for Average Confidence ===")
    ece_conf, brier_conf, nll_conf = compute_calibration_metrics(all_avg_confidences, all_labels)
    print(f"Expected Calibration Error (ECE): {ece_conf:.4f}")
    print(f"Brier Score: {brier_conf:.4f}")
    print(f"Negative Log-Likelihood: {nll_conf:.4f}")
    
    # Compute calibration metrics for evidence quality score
    print("\n=== Calibration Metrics for Evidence Quality Score ===")
    ece_evidence, brier_evidence, nll_evidence = compute_calibration_metrics(all_evidence_scores, all_labels)
    print(f"Expected Calibration Error (ECE): {ece_evidence:.4f}")
    print(f"Brier Score: {brier_evidence:.4f}")
    print(f"Negative Log-Likelihood: {nll_evidence:.4f}")
    
    # Compute overall averages
    print("\n=== Overall Averages ===")
    print(f"Average ECE: {(ece_conf + ece_evidence) / 2:.4f}")
    print(f"Average Brier Score: {(brier_conf + brier_evidence) / 2:.4f}")
    print(f"Average NLL: {(nll_conf + nll_evidence) / 2:.4f}")
    
    # Additional statistics
    print(f"\n=== Additional Statistics ===")
    print(f"Overall accuracy: {sum(all_labels) / len(all_labels):.4f}")
    print(f"Mean average confidence: {sum(all_avg_confidences) / len(all_avg_confidences):.4f}")
    print(f"Mean evidence quality score: {sum(all_evidence_scores) / len(all_evidence_scores):.4f}")
    
    # Compute standard deviations
    mean_conf = sum(all_avg_confidences) / len(all_avg_confidences)
    mean_evidence = sum(all_evidence_scores) / len(all_evidence_scores)
    std_conf = math.sqrt(sum((x - mean_conf) ** 2 for x in all_avg_confidences) / len(all_avg_confidences))
    std_evidence = math.sqrt(sum((x - mean_evidence) ** 2 for x in all_evidence_scores) / len(all_evidence_scores))
    print(f"Std average confidence: {std_conf:.4f}")
    print(f"Std evidence quality score: {std_evidence:.4f}")
    
    # Create calibration plots
    if PLOTTING_AVAILABLE:
        print("\n=== Creating Calibration Plots ===")
        
        # Create output directory for plots
        output_dir = Path("confidence_evaluation_plots")
        output_dir.mkdir(exist_ok=True)
        
        # Plot for average confidence
        create_calibration_plot(
            all_avg_confidences, 
            all_labels, 
            "Average Confidence", 
            output_dir / "average_confidence_calibration.png"
        )
        
        # Plot for evidence quality score
        create_calibration_plot(
            all_evidence_scores, 
            all_labels, 
            "Evidence Quality Score", 
            output_dir / "evidence_quality_calibration.png"
        )
        
        print(f"All plots saved to: {output_dir.absolute()}")
    else:
        print("\n=== Skipping Calibration Plots ===")
        print("Install matplotlib and numpy to generate calibration plots:")
        print("pip install matplotlib numpy")


if __name__ == "__main__":
    main()
