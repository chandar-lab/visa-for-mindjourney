"""
World Model Image Selection Quality Evaluation

This module evaluates the quality of world model images selected by baseline vs verifier systems
by analyzing the selection patterns, image quality metrics, and effectiveness correlations.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re


class WorldModelImageSelectionEvaluator:
    """Evaluates the quality of world model image selection between baseline and verifier systems."""
    
    def __init__(self, baseline_results_path: str, verifier_results_path: str):
        """
        Initialize the evaluator with paths to results directories.
        
        Args:
            baseline_results_path: Path to baseline results directory
            verifier_results_path: Path to verifier results directory
        """
        self.baseline_results_path = baseline_results_path
        self.verifier_results_path = verifier_results_path
        
        # Load results data
        self.baseline_data = self._load_results(baseline_results_path)
        self.verifier_data = self._load_results(verifier_results_path)
        
        # Load individual question data
        self.baseline_questions = self._load_all_questions(baseline_results_path)
        self.verifier_questions = self._load_all_questions(verifier_results_path)
        
    def _load_results(self, results_path: str) -> Dict:
        """Load results from JSON file."""
        results_file = os.path.join(results_path, "results.json")
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def _load_all_questions(self, results_path: str) -> Dict[str, Dict]:
        """Load all individual question data."""
        questions = {}
        # Check for questions 0-99 to be safe
        for qid in range(100):
            gpt_file = os.path.join(results_path, str(qid), "gpt.json")
            if os.path.exists(gpt_file):
                with open(gpt_file, 'r') as f:
                    questions[str(qid)] = json.load(f)
        return questions
    
    def _load_step_data(self, results_path: str, qid: str, step: int = 0) -> Optional[Dict]:
        """Load step-level data for action scoring."""
        step_file = os.path.join(results_path, qid, f"step_{step}", "gpt_0.json")
        if os.path.exists(step_file):
            with open(step_file, 'r') as f:
                return json.load(f)
        return None
    
    def analyze_image_selection_patterns(self) -> Dict[str, Any]:
        """
        Analyze how baseline and verifier systems select world model images.
        
        Returns:
            Dictionary containing image selection analysis
        """
        baseline_selections = self._extract_image_selections(self.baseline_questions, is_verifier=False)
        verifier_selections = self._extract_image_selections(self.verifier_questions, is_verifier=True)
        
        return {
            'baseline_selection_analysis': self._analyze_baseline_selections(baseline_selections),
            'verifier_selection_analysis': self._analyze_verifier_selections(verifier_selections),
            'selection_comparison': self._compare_selection_patterns(baseline_selections, verifier_selections),
            'selection_effectiveness': self._analyze_selection_effectiveness(baseline_selections, verifier_selections)
        }
    
    def _extract_image_selections(self, questions: Dict[str, Dict], is_verifier: bool) -> List[Dict]:
        """Extract image selection data from question results."""
        selections = []
        processed_questions = 0
        questions_with_selections = 0
        
        for qid, question_data in questions.items():
            if not question_data:
                continue
            
            processed_questions += 1
            question_type = question_data.get('question', {}).get('question_type', 'unknown')
            result = question_data.get('result', 'unknown')
            
            # Extract selected images from prompt content
            prompt_content = question_data.get('prompt', {}).get('content', [])
            selected_images = self._parse_selected_images(prompt_content, qid, is_verifier)
            
            if selected_images:
                questions_with_selections += 1
            
            # For baseline, also get scoring data
            baseline_scores = []
            if not is_verifier:
                step_data = self._load_step_data(self.baseline_results_path, qid, 0)
                if step_data and 'llm_response' in step_data:
                    baseline_scores = self._parse_baseline_scores(step_data['llm_response'])
                    all_images = self._extract_all_available_images(step_data)
            
            for i, image_info in enumerate(selected_images):
                selection_data = {
                    'qid': qid,
                    'question_type': question_type,
                    'result': result,
                    'image_path': image_info['path'],
                    'action_type': image_info['action_type'],
                    'magnitude': image_info['magnitude'],
                    'is_verifier': is_verifier
                }
                
                # Add baseline score if available
                if not is_verifier and baseline_scores and i < len(baseline_scores):
                    selection_data['baseline_score'] = baseline_scores[i]
                
                # Add verifier quality metrics if available
                if is_verifier:
                    quality_metrics = self._get_verifier_quality_metrics(qid, image_info)
                    selection_data.update(quality_metrics)
                
                selections.append(selection_data)
        
        # Debug output
        system_name = "Verifier" if is_verifier else "Baseline"
        print(f"{system_name} System: Processed {processed_questions} questions, {questions_with_selections} with selections, {len(selections)} total selections")
        
        return selections
    
    def _parse_selected_images(self, prompt_content: List, qid: str, is_verifier: bool) -> List[Dict]:
        """Parse selected images from prompt content."""
        selected_images = []
        current_action = None
        
        for i, content_block in enumerate(prompt_content):
            if isinstance(content_block, list) and len(content_block) >= 1:
                text = content_block[0] if isinstance(content_block[0], str) else ""
                image_path = content_block[1] if len(content_block) > 1 else ""
                
                # Detect action type - look for "Action:" in text
                if "Action:" in text:
                    current_action = text.replace("Action:", "").strip()
                
                # Check if this is a selected image (has magnitude text and image path)
                elif ("degrees" in text or "meters" in text) and image_path and ("sample_" in image_path or "turn-" in image_path or "move-" in image_path):
                    # Extract magnitude and image info - these are the selected images
                    magnitude = self._extract_magnitude(text)
                    if magnitude is not None and image_path:
                        selected_images.append({
                            'path': image_path,
                            'action_type': current_action,
                            'magnitude': magnitude
                        })
        
        return selected_images
    
    def _extract_magnitude(self, text: str) -> Optional[float]:
        """Extract magnitude value from text."""
        import re
        
        # Look for numbers followed by degrees or meters
        degree_match = re.search(r'(\d+(?:\.\d+)?)\s*degrees', text)
        meter_match = re.search(r'(\d+(?:\.\d+)?)\s*meters', text)
        
        if degree_match:
            return float(degree_match.group(1))
        elif meter_match:
            return float(meter_match.group(1))
        
        return None
    
    def _parse_baseline_scores(self, llm_response: str) -> List[int]:
        """Parse baseline scores from LLM response."""
        try:
            scores = [int(x.strip()) for x in llm_response.split(',') if x.strip().isdigit()]
            return scores
        except (ValueError, AttributeError):
            return []
    
    def _extract_all_available_images(self, step_data: Dict) -> List[Dict]:
        """Extract all available images from step data for baseline comparison."""
        all_images = []
        prompt_content = step_data.get('prompt', {}).get('content', [])
        
        for content_block in prompt_content:
            if isinstance(content_block, list) and len(content_block) >= 2:
                text = content_block[0] if isinstance(content_block[0], str) else ""
                image_path = content_block[1] if len(content_block) > 1 else ""
                
                if "Imagined image of index" in text and image_path:
                    # Extract action info from text
                    action_match = re.search(r'if you (\w+ \w+)', text)
                    if action_match:
                        action_text = action_match.group(1)
                        action_type = action_text.split()[0] + " " + action_text.split()[1]
                        magnitude = self._extract_magnitude(text)
                        
                        all_images.append({
                            'path': image_path,
                            'action_type': action_type,
                            'magnitude': magnitude
                        })
        
        return all_images
    
    def _get_verifier_quality_metrics(self, qid: str, image_info: Dict) -> Dict[str, Any]:
        """Get verifier quality metrics for a specific image."""
        if 'verification_metrics' not in self.verifier_data:
            return {}
        
        verification_metrics = self.verifier_data['verification_metrics']
        qid_data = verification_metrics.get(qid, {})
        
        # Find matching action and magnitude
        for step_data in qid_data.values():
            for action_type, action_data in step_data.items():
                if action_type == image_info['action_type']:
                    for magnitude, metrics in action_data.items():
                        if str(magnitude) == str(image_info['magnitude']):
                            if isinstance(metrics, dict):
                                return {
                                    'consistency_score': metrics.get('consistency_score', 0),
                                    'reliability_score': metrics.get('reliability_score', 0),
                                    'evidence_quality_score': metrics.get('evidence_quality_score', 0),
                                    'helpfulness_score': metrics.get('helpfulness_score', 0),
                                    'exploration_score': metrics.get('exploration_score', 0),
                                    'claim_acceptance_rate': metrics.get('claim_acceptance_rate', 0),
                                    'total_claims': metrics.get('total_claims', 0)
                                }
        
        return {}
    
    def _analyze_baseline_selections(self, selections: List[Dict]) -> Dict[str, Any]:
        """Analyze baseline image selection patterns."""
        if not selections:
            return {'error': 'No baseline selections found'}
        
        # Analyze selection patterns
        action_type_counts = Counter([s['action_type'] for s in selections])
        magnitude_stats = [s['magnitude'] for s in selections if s['magnitude'] is not None]
        
        # Analyze scoring patterns
        scored_selections = [s for s in selections if 'baseline_score' in s]
        scores = [s['baseline_score'] for s in scored_selections]
        
        return {
            'total_selections': len(selections),
            'action_type_distribution': dict(action_type_counts),
            'magnitude_statistics': {
                'mean': np.mean(magnitude_stats) if magnitude_stats else 0,
                'std': np.std(magnitude_stats) if magnitude_stats else 0,
                'median': np.median(magnitude_stats) if magnitude_stats else 0,
                'count': len(magnitude_stats)
            },
            'scoring_analysis': {
                'mean_score': np.mean(scores) if scores else 0,
                'std_score': np.std(scores) if scores else 0,
                'median_score': np.median(scores) if scores else 0,
                'min_score': np.min(scores) if scores else 0,
                'max_score': np.max(scores) if scores else 0,
                'count': len(scores)
            },
            'score_distribution': self._analyze_score_distribution(scores),
            'selection_efficiency': self._analyze_selection_efficiency(selections)
        }
    
    def _analyze_verifier_selections(self, selections: List[Dict]) -> Dict[str, Any]:
        """Analyze verifier image selection patterns."""
        if not selections:
            return {'error': 'No verifier selections found'}
        
        # Analyze selection patterns
        action_type_counts = Counter([s['action_type'] for s in selections])
        magnitude_stats = [s['magnitude'] for s in selections if s['magnitude'] is not None]
        
        # Analyze quality metrics
        quality_metrics = ['consistency_score', 'reliability_score', 'evidence_quality_score', 
                          'helpfulness_score', 'exploration_score', 'claim_acceptance_rate']
        
        quality_analysis = {}
        for metric in quality_metrics:
            values = [s[metric] for s in selections if metric in s and s[metric] is not None]
            if values:
                quality_analysis[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return {
            'total_selections': len(selections),
            'action_type_distribution': dict(action_type_counts),
            'magnitude_statistics': {
                'mean': np.mean(magnitude_stats) if magnitude_stats else 0,
                'std': np.std(magnitude_stats) if magnitude_stats else 0,
                'median': np.median(magnitude_stats) if magnitude_stats else 0,
                'count': len(magnitude_stats)
            },
            'quality_metrics_analysis': quality_analysis,
            'selection_efficiency': self._analyze_selection_efficiency(selections)
        }
    
    def _analyze_score_distribution(self, scores: List[int]) -> Dict[str, Any]:
        """Analyze the distribution of baseline scores."""
        if not scores:
            return {}
        
        score_counts = Counter(scores)
        total = len(scores)
        
        return {
            'score_frequencies': dict(score_counts),
            'score_percentages': {str(k): v/total for k, v in score_counts.items()},
            'high_score_selections': sum(1 for s in scores if s >= 7),
            'low_score_selections': sum(1 for s in scores if s <= 3),
            'medium_score_selections': sum(1 for s in scores if 3 < s < 7)
        }
    
    def _analyze_selection_efficiency(self, selections: List[Dict]) -> Dict[str, Any]:
        """Analyze selection efficiency metrics."""
        if not selections:
            return {}
        
        # Group by question
        by_question = defaultdict(list)
        for selection in selections:
            by_question[selection['qid']].append(selection)
        
        # Calculate efficiency metrics
        images_per_question = [len(selections) for selections in by_question.values()]
        correct_questions = sum(1 for qid, q_selections in by_question.items() 
                              if any(s['result'] == 'correct' for s in q_selections))
        
        return {
            'avg_images_per_question': np.mean(images_per_question),
            'std_images_per_question': np.std(images_per_question),
            'max_images_per_question': np.max(images_per_question),
            'min_images_per_question': np.min(images_per_question),
            'questions_with_selections': len(by_question),
            'correct_questions': correct_questions,
            'selection_success_rate': correct_questions / len(by_question) if by_question else 0
        }
    
    def _compare_selection_patterns(self, baseline_selections: List[Dict], verifier_selections: List[Dict]) -> Dict[str, Any]:
        """Compare selection patterns between baseline and verifier systems."""
        return {
            'selection_count_comparison': {
                'baseline_total': len(baseline_selections),
                'verifier_total': len(verifier_selections),
                'difference': len(verifier_selections) - len(baseline_selections),
                'ratio': len(verifier_selections) / len(baseline_selections) if baseline_selections else 0
            },
            'action_type_comparison': self._compare_action_type_selections(baseline_selections, verifier_selections),
            'magnitude_comparison': self._compare_magnitude_selections(baseline_selections, verifier_selections),
            'selection_diversity': self._compare_selection_diversity(baseline_selections, verifier_selections)
        }
    
    def _compare_action_type_selections(self, baseline_selections: List[Dict], verifier_selections: List[Dict]) -> Dict[str, Any]:
        """Compare action type selection patterns."""
        baseline_types = Counter([s['action_type'] for s in baseline_selections])
        verifier_types = Counter([s['action_type'] for s in verifier_selections])
        
        all_types = set(list(baseline_types.keys()) + list(verifier_types.keys()))
        
        comparison = {}
        for action_type in all_types:
            baseline_count = baseline_types.get(action_type, 0)
            verifier_count = verifier_types.get(action_type, 0)
            
            comparison[action_type] = {
                'baseline_count': baseline_count,
                'verifier_count': verifier_count,
                'difference': verifier_count - baseline_count,
                'relative_change': (verifier_count - baseline_count) / baseline_count if baseline_count > 0 else float('inf')
            }
        
        return comparison
    
    def _compare_magnitude_selections(self, baseline_selections: List[Dict], verifier_selections: List[Dict]) -> Dict[str, Any]:
        """Compare magnitude selection patterns."""
        baseline_magnitudes = [s['magnitude'] for s in baseline_selections if s['magnitude'] is not None]
        verifier_magnitudes = [s['magnitude'] for s in verifier_selections if s['magnitude'] is not None]
        
        return {
            'baseline_stats': {
                'mean': np.mean(baseline_magnitudes) if baseline_magnitudes else 0,
                'std': np.std(baseline_magnitudes) if baseline_magnitudes else 0,
                'median': np.median(baseline_magnitudes) if baseline_magnitudes else 0,
                'count': len(baseline_magnitudes)
            },
            'verifier_stats': {
                'mean': np.mean(verifier_magnitudes) if verifier_magnitudes else 0,
                'std': np.std(verifier_magnitudes) if verifier_magnitudes else 0,
                'median': np.median(verifier_magnitudes) if verifier_magnitudes else 0,
                'count': len(verifier_magnitudes)
            },
            'statistical_significance': self._test_magnitude_differences(baseline_magnitudes, verifier_magnitudes)
        }
    
    def _test_magnitude_differences(self, baseline_mags: List[float], verifier_mags: List[float]) -> Dict[str, Any]:
        """Test statistical significance of magnitude differences."""
        if not baseline_mags or not verifier_mags:
            return {'error': 'Insufficient data for statistical tests'}
        
        try:
            from scipy import stats
            
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(baseline_mags, verifier_mags, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_mags) - 1) * np.var(baseline_mags) + 
                                 (len(verifier_mags) - 1) * np.var(verifier_mags)) / 
                                (len(baseline_magnitudes) + len(verifier_magnitudes) - 2))
            cohens_d = (np.mean(verifier_mags) - np.mean(baseline_mags)) / pooled_std if pooled_std > 0 else 0
            
            return {
                'mann_whitney_u': u_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohens_d': cohens_d,
                'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
            }
        except Exception as e:
            return {'error': f'Statistical test failed: {str(e)}'}
    
    def _compare_selection_diversity(self, baseline_selections: List[Dict], verifier_selections: List[Dict]) -> Dict[str, Any]:
        """Compare selection diversity between systems."""
        # Calculate diversity metrics
        baseline_diversity = self._calculate_selection_diversity(baseline_selections)
        verifier_diversity = self._calculate_selection_diversity(verifier_selections)
        
        # Handle empty diversity results
        baseline_entropy = baseline_diversity.get('action_type_entropy', 0)
        verifier_entropy = verifier_diversity.get('action_type_entropy', 0)
        baseline_variance = baseline_diversity.get('magnitude_variance', 0)
        verifier_variance = verifier_diversity.get('magnitude_variance', 0)
        baseline_combinations = baseline_diversity.get('unique_combinations', 0)
        verifier_combinations = verifier_diversity.get('unique_combinations', 0)
        
        return {
            'baseline_diversity': baseline_diversity,
            'verifier_diversity': verifier_diversity,
            'diversity_comparison': {
                'action_type_entropy_diff': verifier_entropy - baseline_entropy,
                'magnitude_variance_diff': verifier_variance - baseline_variance,
                'unique_combinations_diff': verifier_combinations - baseline_combinations
            }
        }
    
    def _calculate_selection_diversity(self, selections: List[Dict]) -> Dict[str, Any]:
        """Calculate diversity metrics for selections."""
        if not selections:
            return {}
        
        # Action type diversity
        action_types = [s['action_type'] for s in selections]
        action_type_counts = Counter(action_types)
        action_type_entropy = self._calculate_entropy(action_type_counts)
        
        # Magnitude diversity
        magnitudes = [s['magnitude'] for s in selections if s['magnitude'] is not None]
        magnitude_variance = np.var(magnitudes) if magnitudes else 0
        
        # Unique combinations
        combinations = [(s['action_type'], s['magnitude']) for s in selections]
        unique_combinations = len(set(combinations))
        
        return {
            'action_type_entropy': action_type_entropy,
            'magnitude_variance': magnitude_variance,
            'unique_combinations': unique_combinations,
            'total_selections': len(selections)
        }
    
    def _calculate_entropy(self, counts: Counter) -> float:
        """Calculate Shannon entropy of a distribution."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _analyze_selection_effectiveness(self, baseline_selections: List[Dict], verifier_selections: List[Dict]) -> Dict[str, Any]:
        """Analyze the effectiveness of image selections."""
        return {
            'baseline_effectiveness': self._calculate_selection_effectiveness(baseline_selections),
            'verifier_effectiveness': self._calculate_selection_effectiveness(verifier_selections),
            'effectiveness_comparison': self._compare_selection_effectiveness(baseline_selections, verifier_selections),
            'quality_correlation_analysis': self._analyze_quality_correlations(verifier_selections)
        }
    
    def _calculate_selection_effectiveness(self, selections: List[Dict]) -> Dict[str, Any]:
        """Calculate effectiveness metrics for selections."""
        if not selections:
            return {}
        
        # Group by question
        by_question = defaultdict(list)
        for selection in selections:
            by_question[selection['qid']].append(selection)
        
        # Calculate effectiveness metrics
        total_questions = len(by_question)
        correct_questions = sum(1 for qid, q_selections in by_question.items() 
                              if any(s['result'] == 'correct' for s in q_selections))
        
        # Calculate per-action effectiveness
        action_effectiveness = defaultdict(lambda: {'correct': 0, 'total': 0})
        for selection in selections:
            action_type = selection['action_type']
            action_effectiveness[action_type]['total'] += 1
            if selection['result'] == 'correct':
                action_effectiveness[action_type]['correct'] += 1
        
        # Convert to rates
        action_rates = {}
        for action_type, counts in action_effectiveness.items():
            if counts['total'] > 0:
                action_rates[action_type] = counts['correct'] / counts['total']
            else:
                action_rates[action_type] = 0.0
        
        return {
            'overall_success_rate': correct_questions / total_questions if total_questions > 0 else 0,
            'total_questions': total_questions,
            'correct_questions': correct_questions,
            'action_type_effectiveness': action_rates,
            'avg_selections_per_question': len(selections) / total_questions if total_questions > 0 else 0
        }
    
    def _compare_selection_effectiveness(self, baseline_selections: List[Dict], verifier_selections: List[Dict]) -> Dict[str, Any]:
        """Compare effectiveness between baseline and verifier selections."""
        baseline_eff = self._calculate_selection_effectiveness(baseline_selections)
        verifier_eff = self._calculate_selection_effectiveness(verifier_selections)
        
        # Handle empty effectiveness results
        baseline_success_rate = baseline_eff.get('overall_success_rate', 0)
        verifier_success_rate = verifier_eff.get('overall_success_rate', 0)
        baseline_avg_selections = baseline_eff.get('avg_selections_per_question', 0)
        verifier_avg_selections = verifier_eff.get('avg_selections_per_question', 0)
        
        return {
            'success_rate_improvement': verifier_success_rate - baseline_success_rate,
            'relative_improvement': (verifier_success_rate - baseline_success_rate) / baseline_success_rate if baseline_success_rate > 0 else 0,
            'efficiency_comparison': {
                'baseline_avg_selections': baseline_avg_selections,
                'verifier_avg_selections': verifier_avg_selections,
                'efficiency_ratio': verifier_avg_selections / baseline_avg_selections if baseline_avg_selections > 0 else 0
            }
        }
    
    def _analyze_quality_correlations(self, verifier_selections: List[Dict]) -> Dict[str, Any]:
        """Analyze correlations between quality metrics and effectiveness."""
        if not verifier_selections:
            return {}
        
        quality_metrics = ['consistency_score', 'reliability_score', 'evidence_quality_score', 
                          'helpfulness_score', 'exploration_score', 'claim_acceptance_rate']
        
        correlations = {}
        for metric in quality_metrics:
            values = []
            correctness = []
            
            for selection in verifier_selections:
                if metric in selection and selection[metric] is not None:
                    values.append(selection[metric])
                    correctness.append(1 if selection['result'] == 'correct' else 0)
            
            if len(values) > 1:
                correlation = np.corrcoef(values, correctness)[0, 1]
                correlations[metric] = {
                    'correlation': correlation,
                    'sample_size': len(values),
                    'interpretation': 'strong positive' if correlation > 0.7 else 'moderate positive' if correlation > 0.3 else 'weak positive' if correlation > 0.1 else 'weak negative' if correlation > -0.1 else 'moderate negative' if correlation > -0.3 else 'strong negative'
                }
        
        return correlations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of image selection quality analysis.
        
        Returns:
            Dictionary containing complete analysis report
        """
        report = {
            'image_selection_patterns': self.analyze_image_selection_patterns(),
            'summary_statistics': self._generate_summary_statistics()
        }
        
        return report
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the analysis."""
        baseline_accuracy = self.baseline_data.get('accuracy', {}).get('all', 0)
        verifier_accuracy = self.verifier_data.get('accuracy', {}).get('all', 0)
        
        # Count questions with actual data
        baseline_questions_with_data = len([q for q in self.baseline_questions.values() if q])
        verifier_questions_with_data = len([q for q in self.verifier_questions.values() if q])
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'verifier_accuracy': verifier_accuracy,
            'accuracy_improvement': verifier_accuracy - baseline_accuracy,
            'relative_improvement': (verifier_accuracy - baseline_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0,
            'baseline_questions_loaded': len(self.baseline_questions),
            'verifier_questions_loaded': len(self.verifier_questions),
            'baseline_questions_with_data': baseline_questions_with_data,
            'verifier_questions_with_data': verifier_questions_with_data,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def save_report(self, output_path: str) -> None:
        """Save the comprehensive report to a JSON file."""
        report = self.generate_comprehensive_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"World model image selection quality report saved to: {output_path}")


def main():
    """Main function to run the image selection quality analysis."""
    # Define paths
    baseline_path = "/Users/saurav/Documents/MindJourney/results/svc_test_o4-mini_150_1_2_2_2_spatial_beam_search"
    verifier_path = "/Users/saurav/Documents/MindJourney/results/svc_test_o4-mini_150_1_2_2_2_with_verifier_spatial_beam_search"
    
    # Initialize evaluator
    evaluator = WorldModelImageSelectionEvaluator(baseline_path, verifier_path)
    
    # Generate and save report
    output_file = "/Users/saurav/Documents/MindJourney/utils/further_evals/world_model_image_selection_quality_report_reliablity_ablation.json"
    evaluator.save_report(output_file)
    
    # Print summary
    report = evaluator.generate_comprehensive_report()
    summary = report['summary_statistics']
    
    print("\n" + "="*70)
    print("WORLD MODEL IMAGE SELECTION QUALITY ANALYSIS SUMMARY")
    print("="*70)
    print(f"Baseline Accuracy: {summary['baseline_accuracy']:.3f}")
    print(f"Verifier Accuracy: {summary['verifier_accuracy']:.3f}")
    print(f"Accuracy Improvement: {summary['accuracy_improvement']:.3f}")
    print(f"Relative Improvement: {summary['relative_improvement']:.1%}")
    print(f"Baseline Questions Loaded: {summary['baseline_questions_loaded']}")
    print(f"Verifier Questions Loaded: {summary['verifier_questions_loaded']}")
    print(f"Baseline Questions with Data: {summary['baseline_questions_with_data']}")
    print(f"Verifier Questions with Data: {summary['verifier_questions_with_data']}")
    print("="*70)


if __name__ == "__main__":
    main()
