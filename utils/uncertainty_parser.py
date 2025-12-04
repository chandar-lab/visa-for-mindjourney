"""
Uncertainty metrics parsing utility for VQA results.

This module provides functionality to parse uncertainty metrics from result directories
containing VQA evaluation results. It extracts and averages uncertainty metrics
(log_probs, token_entropy, answer_entropy) across all questions and separately
for correct and incorrect answers.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import statistics


def parse_uncertainty_metrics(result_folder_path: str) -> Dict[str, Any]:
    """
    Parse uncertainty metrics from a result directory containing VQA evaluation results.
    
    Args:
        result_folder_path (str): Path to the result directory containing numbered subfolders
                                 with gpt.json files containing uncertainty metrics.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'overall': Average metrics across all questions
            - 'correct': Average metrics for correct answers only
            - 'wrong': Average metrics for wrong answers only
            - 'summary': Summary statistics including counts and standard deviations
    
    Raises:
        FileNotFoundError: If the result folder path doesn't exist
        ValueError: If no valid gpt.json files are found
    """
    
    result_path = Path(result_folder_path)
    if not result_path.exists():
        raise FileNotFoundError(f"Result folder not found: {result_folder_path}")
    
    # Collect all numbered subdirectories
    question_dirs = []
    for item in result_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            question_dirs.append(item)
    
    if not question_dirs:
        raise ValueError(f"No numbered subdirectories found in {result_folder_path}")
    
    # Parse all gpt.json files
    all_metrics = []
    correct_metrics = []
    wrong_metrics = []
    
    num_generated_answers = 0
    num_non_generated_answers = 0
    for question_dir in sorted(question_dirs, key=lambda x: int(x.name)):
        gpt_file = question_dir / "gpt.json"
        
        if not gpt_file.exists():
            gpt_file = question_dir / "step_0" / "gpt.json"
            print(f"Warning: gpt.json not found in {question_dir}, next trying step_0/gpt.json")
            if not gpt_file.exists():
                print(f"Warning: gpt.json not found in {question_dir / 'step_0'}/ either")
                continue
            
        try:
            with open(gpt_file, 'r') as f:
                data = json.load(f)
            
            # Extract uncertainty metrics
            if 'uncertainty_metrics' not in data:
                print(f"Warning: uncertainty_metrics not found in {gpt_file}")
                continue
                
            uncertainty_metrics = data['uncertainty_metrics']
            result = data.get('result', 'unknown')
            
            # Extract metrics for each answer choice
            log_probs = uncertainty_metrics.get('log_probs', {})
            token_entropy = uncertainty_metrics.get('token_entropy', {})
            answer_entropy = uncertainty_metrics.get('answer_entropy', 0.0)
            
            # Determine the generated answer and non-generated answer
            answer_choices = data['question']['answer_choices']
            correct_answer = data['question']['correct_answer']
            
            if result == 'correct':
                generated_answer = correct_answer
            elif result == 'wrong':
                # Find the other choice (the generated answer)
                generated_answer = None
                for choice in answer_choices:
                    if choice != correct_answer:
                        generated_answer = choice
                        break
                if generated_answer is None:
                    print(f"Warning: Could not determine generated answer for {gpt_file}")
                    continue
            else:
                print(f"Warning: Unknown result '{result}' in {gpt_file}")
                continue
            
            # Determine the non-generated answer (the alternative choice)
            non_generated_answer = None
            for choice in answer_choices:
                if choice != generated_answer:
                    non_generated_answer = choice
                    num_non_generated_answers += 1
                    break
            if non_generated_answer is None:
                print(f"Warning: Could not determine non-generated answer for {gpt_file}")
                continue
            
            # Calculate average log_probs and token_entropy across answer choices
            avg_log_prob = statistics.mean(log_probs.values()) if log_probs else 0.0
            avg_token_entropy = statistics.mean(token_entropy.values()) if token_entropy else 0.0
            
            # Extract metrics for the generated answer specifically
            generated_log_prob = log_probs.get(generated_answer, 0.0)
            generated_token_entropy = token_entropy.get(generated_answer, 0.0)
            
            # Extract metrics for the non-generated answer specifically
            non_generated_log_prob = log_probs.get(non_generated_answer, 0.0)
            non_generated_token_entropy = token_entropy.get(non_generated_answer, 0.0)
            
            metrics_entry = {
                'question_id': question_dir.name,
                'avg_log_prob': avg_log_prob,
                'avg_token_entropy': avg_token_entropy,
                'answer_entropy': answer_entropy,
                'generated_answer': generated_answer,
                'generated_log_prob': generated_log_prob,
                'generated_token_entropy': generated_token_entropy,
                'non_generated_answer': non_generated_answer,
                'non_generated_log_prob': non_generated_log_prob,
                'non_generated_token_entropy': non_generated_token_entropy,
                'result': result
            }
            
            all_metrics.append(metrics_entry)
            
            if result == 'correct':
                correct_metrics.append(metrics_entry)
            elif result == 'wrong':
                wrong_metrics.append(metrics_entry)
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing {gpt_file}: {e}")
            continue
    
    if not all_metrics:
        raise ValueError("No valid uncertainty metrics found in any gpt.json files")
    
    # Calculate averages
    def calculate_averages(metrics_list: List[Dict]) -> Dict[str, float]:
        if not metrics_list:
            return {
                'avg_log_prob': 0.0,
                'avg_token_entropy': 0.0,
                'avg_answer_entropy': 0.0,
                'avg_generated_log_prob': 0.0,
                'avg_generated_token_entropy': 0.0,
                'avg_non_generated_log_prob': 0.0,
                'avg_non_generated_token_entropy': 0.0,
                'count': 0
            }
        
        return {
            'avg_log_prob': statistics.mean([m['avg_log_prob'] for m in metrics_list]),
            'avg_token_entropy': statistics.mean([m['avg_token_entropy'] for m in metrics_list]),
            'avg_answer_entropy': statistics.mean([m['answer_entropy'] for m in metrics_list]),
            'avg_generated_log_prob': statistics.mean([m['generated_log_prob'] for m in metrics_list]),
            'avg_generated_token_entropy': statistics.mean([m['generated_token_entropy'] for m in metrics_list]),
            'avg_non_generated_log_prob': statistics.mean([m['non_generated_log_prob'] for m in metrics_list]),
            'avg_non_generated_token_entropy': statistics.mean([m['non_generated_token_entropy'] for m in metrics_list]),
            'count': len(metrics_list)
        }
    
    def calculate_std_devs(metrics_list: List[Dict]) -> Dict[str, float]:
        if len(metrics_list) < 2:
            return {
                'std_log_prob': 0.0,
                'std_token_entropy': 0.0,
                'std_answer_entropy': 0.0,
                'std_generated_log_prob': 0.0,
                'std_generated_token_entropy': 0.0,
                'std_non_generated_log_prob': 0.0,
                'std_non_generated_token_entropy': 0.0
            }
        
        return {
            'std_log_prob': statistics.stdev([m['avg_log_prob'] for m in metrics_list]),
            'std_token_entropy': statistics.stdev([m['avg_token_entropy'] for m in metrics_list]),
            'std_answer_entropy': statistics.stdev([m['answer_entropy'] for m in metrics_list]),
            'std_generated_log_prob': statistics.stdev([m['generated_log_prob'] for m in metrics_list]),
            'std_generated_token_entropy': statistics.stdev([m['generated_token_entropy'] for m in metrics_list]),
            'std_non_generated_log_prob': statistics.stdev([m['non_generated_log_prob'] for m in metrics_list]),
            'std_non_generated_token_entropy': statistics.stdev([m['non_generated_token_entropy'] for m in metrics_list])
        }
    
    # Calculate results
    overall_avg = calculate_averages(all_metrics)
    correct_avg = calculate_averages(correct_metrics)
    wrong_avg = calculate_averages(wrong_metrics)
    
    overall_std = calculate_std_devs(all_metrics)
    correct_std = calculate_std_devs(correct_metrics)
    wrong_std = calculate_std_devs(wrong_metrics)
    
    return {
        'overall': {
            **overall_avg,
            # **overall_std
        },
        'correct': {
            **correct_avg,
            # **correct_std
        },
        'wrong': {
            **wrong_avg,
            # **wrong_std
        },
        'generated_answers': {
            **overall_avg,  # Same as overall since we're looking at all generated answers
            # **overall_std
        },
        'non_generated_answers': {
            **overall_avg,  # Same as overall since we're looking at all non-generated answers
            # **overall_std
        },
        'summary': {
            'total_questions': len(all_metrics),
            'correct_questions': len(correct_metrics),
            'wrong_questions': len(wrong_metrics),
            'accuracy': len(correct_metrics) / len(all_metrics) if all_metrics else 0.0
        }
    }


def print_uncertainty_report(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted report of uncertainty metrics.
    
    Args:
        metrics (Dict[str, Any]): Output from parse_uncertainty_metrics function
    """
    print("=" * 80)
    print("UNCERTAINTY METRICS REPORT")
    print("=" * 80)
    
    summary = metrics['summary']
    print(f"\nSUMMARY:")
    print(f"  Total Questions: {summary['total_questions']}")
    print(f"  Correct Answers: {summary['correct_questions']}")
    print(f"  Wrong Answers: {summary['wrong_questions']}")
    print(f"  Accuracy: {summary['accuracy']:.3f}")
    
    def print_metrics_section(title: str, data: Dict[str, float], show_generated: bool = False, show_non_generated: bool = False) -> None:
        print(f"\n{title.upper()}:")
        if show_generated:
            print(f"  Generated Log Probability: {data['avg_generated_log_prob']:.4f} ± {data['std_generated_log_prob']:.4f}")
            print(f"  Generated Token Entropy:   {data['avg_generated_token_entropy']:.4f} ± {data['std_generated_token_entropy']:.4f}")
            print(f"  Non-Generated Log Probability: {data['avg_non_generated_log_prob']:.4f} ± {data['std_non_generated_log_prob']:.4f}")
            print(f"  Non-Generated Token Entropy:   {data['avg_non_generated_token_entropy']:.4f} ± {data['std_non_generated_token_entropy']:.4f}")
        else:
            print(f"  Average Log Probability: {data['avg_log_prob']:.4f} ± {data['std_log_prob']:.4f}")
            print(f"  Average Token Entropy:   {data['avg_token_entropy']:.4f} ± {data['std_token_entropy']:.4f}")
            print(f"  Average Answer Entropy:  {data['avg_answer_entropy']:.4f} ± {data['std_answer_entropy']:.4f}")
        print(f"  Count: {data['count']}")
    
    print_metrics_section("Overall Metrics", metrics['overall'])
    print_metrics_section("Correct Answers", metrics['correct'])
    print_metrics_section("Wrong Answers", metrics['wrong'])
    print_metrics_section("Generated Answers", metrics['generated_answers'], show_generated=True)
    # print_metrics_section("Non-Generated Answers", metrics['non_generated_answers'], show_non_generated=True)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python uncertainty_parser.py <result_folder_path>")
        sys.exit(1)
    
    result_folder = sys.argv[1]
    
    try:
        metrics = parse_uncertainty_metrics(result_folder)
        print_uncertainty_report(metrics)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
