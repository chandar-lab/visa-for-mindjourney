"""
World Model Utilization Metrics for MindJourney Results

This module provides comprehensive analysis of world model utilization and quality
metrics from the baseline and verifier systems, focusing on action selection,
world model quality, and utilization effectiveness.
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


class WorldModelUtilizationAnalyzer:
    """Analyzes world model utilization and quality metrics from MindJourney results."""
    
    def __init__(self, baseline_results_path: str, verifier_results_path: str):
        """
        Initialize the analyzer with paths to results directories.
        
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
        for qid in range(62):  # Assuming 50 questions
            gpt_file = os.path.join(results_path, str(qid), "gpt.json")
            if os.path.exists(gpt_file):
                with open(gpt_file, 'r') as f:
                    questions[str(qid)] = json.load(f)
            else:
                print(f"No GPT file found for question {gpt_file}")
        return questions
    
    def _load_step_data(self, results_path: str, qid: str, step: int = 0) -> Optional[Dict]:
        """Load step-level data for action scoring."""
        step_file = os.path.join(results_path, qid, f"step_{step}", "gpt_0.json")
        if os.path.exists(step_file):
            with open(step_file, 'r') as f:
                return json.load(f)
        return None
    
    def analyze_action_selection_quality(self) -> Dict[str, Any]:
        """
        Analyze action selection quality metrics comparing baseline vs verifier.
        
        Returns:
            Dictionary containing action selection analysis results
        """
        baseline_actions = self._extract_action_data(self.baseline_questions, is_verifier=False)
        verifier_actions = self._extract_action_data(self.verifier_questions, is_verifier=True)
        
        results = {
            'action_diversity': self._analyze_action_diversity(baseline_actions, verifier_actions),
            'action_magnitude_optimization': self._analyze_action_magnitudes(baseline_actions, verifier_actions),
            'action_question_alignment': self._analyze_action_question_alignment(baseline_actions, verifier_actions),
            'action_effectiveness': self._analyze_action_effectiveness(baseline_actions, verifier_actions),
            'baseline_vs_verifier_comparison': self._compare_baseline_verifier_actions(baseline_actions, verifier_actions)
        }
        
        return results
    
    def _extract_action_data(self, questions: Dict[str, Dict], is_verifier: bool) -> List[Dict]:
        """Extract action data from question results."""
        actions = []
        
        for qid, question_data in questions.items():
            if not question_data:
                continue
                
            question_type = question_data.get('question', {}).get('question_type', 'unknown')
            result = question_data.get('result', 'unknown')
            magnitude = question_data.get('magnitude')
            
            # Extract actions from prompt content
            prompt_content = question_data.get('prompt', {}).get('content', [])
            extracted_actions = self._parse_actions_from_prompt(prompt_content)
            
            # For baseline, also try to get scoring data from step files
            baseline_scores = []
            if not is_verifier:
                step_data = self._load_step_data(self.baseline_results_path, qid, 0)
                if step_data and 'llm_response' in step_data:
                    baseline_scores = self._parse_baseline_scores(step_data['llm_response'])
            
            for i, action in enumerate(extracted_actions):
                action_data = {
                    'qid': qid,
                    'question_type': question_type,
                    'result': result,
                    'action_type': action['type'],
                    'magnitude': action['magnitude'],
                    'is_verifier': is_verifier
                }
                
                # Add baseline score if available
                if baseline_scores and i < len(baseline_scores):
                    action_data['baseline_score'] = baseline_scores[i]
                
                actions.append(action_data)
        
        return actions
    
    def _parse_actions_from_prompt(self, prompt_content: List) -> List[Dict]:
        """Parse action information from prompt content."""
        actions = []
        current_action = None
        
        for content_block in prompt_content:
            if isinstance(content_block, list) and len(content_block) >= 1:
                text = content_block[0] if isinstance(content_block[0], str) else ""
                
                # Detect action type
                if "Action:" in text:
                    current_action = text.replace("Action:", "").strip()
                elif current_action and ("degrees" in text or "meters" in text):
                    # Extract magnitude
                    magnitude = self._extract_magnitude(text)
                    if magnitude is not None:
                        actions.append({
                            'type': current_action,
                            'magnitude': magnitude
                        })
        
        return actions
    
    def _parse_baseline_scores(self, llm_response: str) -> List[int]:
        """Parse baseline scores from LLM response."""
        try:
            # Extract scores from response like "2,2,3,1,1,1,5,7,9"
            scores = [int(x.strip()) for x in llm_response.split(',') if x.strip().isdigit()]
            return scores
        except (ValueError, AttributeError):
            return []
    
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
    
    def _analyze_action_diversity(self, baseline_actions: List[Dict], verifier_actions: List[Dict]) -> Dict[str, Any]:
        """Analyze action diversity between systems."""
        baseline_types = [a['action_type'] for a in baseline_actions]
        verifier_types = [a['action_type'] for a in verifier_actions]
        
        baseline_counts = Counter(baseline_types)
        verifier_counts = Counter(verifier_types)
        
        return {
            'baseline_distribution': dict(baseline_counts),
            'verifier_distribution': dict(verifier_counts),
            'baseline_unique_actions': len(set(baseline_types)),
            'verifier_unique_actions': len(set(verifier_types)),
            'diversity_entropy': {
                'baseline': self._calculate_entropy(baseline_counts),
                'verifier': self._calculate_entropy(verifier_counts)
            }
        }
    
    def _calculate_entropy(self, counts: Counter) -> float:
        """Calculate Shannon entropy of action distribution."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _analyze_action_magnitudes(self, baseline_actions: List[Dict], verifier_actions: List[Dict]) -> Dict[str, Any]:
        """Analyze action magnitude optimization."""
        baseline_magnitudes = [a['magnitude'] for a in baseline_actions if a['magnitude'] is not None]
        verifier_magnitudes = [a['magnitude'] for a in verifier_actions if a['magnitude'] is not None]
        
        return {
            'baseline_stats': {
                'mean': np.mean(baseline_magnitudes) if baseline_magnitudes else 0,
                'std': np.std(baseline_magnitudes) if baseline_magnitudes else 0,
                'median': np.median(baseline_magnitudes) if baseline_magnitudes else 0,
                'range': (np.min(baseline_magnitudes), np.max(baseline_magnitudes)) if baseline_magnitudes else (0, 0)
            },
            'verifier_stats': {
                'mean': np.mean(verifier_magnitudes) if verifier_magnitudes else 0,
                'std': np.std(verifier_magnitudes) if verifier_magnitudes else 0,
                'median': np.median(verifier_magnitudes) if verifier_magnitudes else 0,
                'range': (np.min(verifier_magnitudes), np.max(verifier_magnitudes)) if verifier_magnitudes else (0, 0)
            },
            'magnitude_by_action_type': self._analyze_magnitude_by_action_type(baseline_actions, verifier_actions)
        }
    
    def _analyze_magnitude_by_action_type(self, baseline_actions: List[Dict], verifier_actions: List[Dict]) -> Dict[str, Any]:
        """Analyze magnitude distribution by action type."""
        baseline_by_type = defaultdict(list)
        verifier_by_type = defaultdict(list)
        
        for action in baseline_actions:
            if action['magnitude'] is not None:
                baseline_by_type[action['action_type']].append(action['magnitude'])
        
        for action in verifier_actions:
            if action['magnitude'] is not None:
                verifier_by_type[action['action_type']].append(action['magnitude'])
        
        result = {}
        for action_type in set(list(baseline_by_type.keys()) + list(verifier_by_type.keys())):
            baseline_mags = baseline_by_type[action_type]
            verifier_mags = verifier_by_type[action_type]
            
            result[action_type] = {
                'baseline_mean': np.mean(baseline_mags) if baseline_mags else 0,
                'verifier_mean': np.mean(verifier_mags) if verifier_mags else 0,
                'baseline_std': np.std(baseline_mags) if baseline_mags else 0,
                'verifier_std': np.std(verifier_mags) if verifier_mags else 0
            }
        
        return result
    
    def _analyze_action_question_alignment(self, baseline_actions: List[Dict], verifier_actions: List[Dict]) -> Dict[str, Any]:
        """Analyze how well actions align with question types."""
        baseline_alignment = self._calculate_action_question_alignment(baseline_actions)
        verifier_alignment = self._calculate_action_question_alignment(verifier_actions)
        
        return {
            'baseline_alignment': baseline_alignment,
            'verifier_alignment': verifier_alignment,
            'alignment_improvement': self._calculate_alignment_improvement(baseline_alignment, verifier_alignment)
        }
    
    def _calculate_action_question_alignment(self, actions: List[Dict]) -> Dict[str, Any]:
        """Calculate action-question type alignment scores."""
        alignment_scores = defaultdict(list)
        
        for action in actions:
            question_type = action['question_type']
            action_type = action['action_type']
            result = action['result']
            
            # Simple alignment scoring based on question type and action type
            score = self._get_alignment_score(question_type, action_type, result)
            alignment_scores[question_type].append(score)
        
        # Calculate average alignment per question type
        avg_alignment = {}
        for qtype, scores in alignment_scores.items():
            avg_alignment[qtype] = {
                'mean_score': np.mean(scores) if scores else 0,
                'std_score': np.std(scores) if scores else 0,
                'count': len(scores)
            }
        
        return avg_alignment
    
    def _get_alignment_score(self, question_type: str, action_type: str, result: str) -> float:
        """Calculate alignment score between question type and action type."""
        # Define expected action types for each question type
        expected_actions = {
            'perspective': ['turn left', 'turn right'],
            'ego_movement': ['move forward', 'turn left', 'turn right'],
            'goal_aim': ['turn left', 'turn right'],
            'obj_movement': ['move forward', 'turn left', 'turn right'],
            'action_conseq': ['move forward', 'turn left', 'turn right']
        }
        
        expected = expected_actions.get(question_type, [])
        is_expected = action_type in expected
        
        # Base score on whether action is expected for question type
        base_score = 1.0 if is_expected else 0.5
        
        # Adjust based on result correctness
        if result == 'correct':
            return base_score
        elif result == 'wrong':
            return base_score * 0.5
        else:
            return base_score * 0.8
    
    def _calculate_alignment_improvement(self, baseline_alignment: Dict, verifier_alignment: Dict) -> Dict[str, float]:
        """Calculate improvement in alignment scores."""
        improvement = {}
        
        for qtype in set(list(baseline_alignment.keys()) + list(verifier_alignment.keys())):
            baseline_score = baseline_alignment.get(qtype, {}).get('mean_score', 0)
            verifier_score = verifier_alignment.get(qtype, {}).get('mean_score', 0)
            
            if baseline_score > 0:
                improvement[qtype] = (verifier_score - baseline_score) / baseline_score
            else:
                improvement[qtype] = verifier_score - baseline_score
        
        return improvement
    
    def _analyze_action_effectiveness(self, baseline_actions: List[Dict], verifier_actions: List[Dict]) -> Dict[str, Any]:
        """Analyze action effectiveness in leading to correct answers."""
        baseline_effectiveness = self._calculate_action_effectiveness(baseline_actions)
        verifier_effectiveness = self._calculate_action_effectiveness(verifier_actions)
        
        return {
            'baseline_effectiveness': baseline_effectiveness,
            'verifier_effectiveness': verifier_effectiveness,
            'effectiveness_improvement': self._calculate_effectiveness_improvement(baseline_effectiveness, verifier_effectiveness)
        }
    
    def _calculate_action_effectiveness(self, actions: List[Dict]) -> Dict[str, float]:
        """Calculate effectiveness of different action types."""
        action_results = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for action in actions:
            action_type = action['action_type']
            result = action['result']
            
            action_results[action_type]['total'] += 1
            if result == 'correct':
                action_results[action_type]['correct'] += 1
        
        effectiveness = {}
        for action_type, counts in action_results.items():
            if counts['total'] > 0:
                effectiveness[action_type] = counts['correct'] / counts['total']
            else:
                effectiveness[action_type] = 0.0
        
        return effectiveness
    
    def _calculate_effectiveness_improvement(self, baseline_eff: Dict, verifier_eff: Dict) -> Dict[str, float]:
        """Calculate improvement in action effectiveness."""
        improvement = {}
        
        for action_type in set(list(baseline_eff.keys()) + list(verifier_eff.keys())):
            baseline_score = baseline_eff.get(action_type, 0)
            verifier_score = verifier_eff.get(action_type, 0)
            
            if baseline_score > 0:
                improvement[action_type] = (verifier_score - baseline_score) / baseline_score
            else:
                improvement[action_type] = verifier_score - baseline_score
        
        return improvement
    
    def _compare_baseline_verifier_actions(self, baseline_actions: List[Dict], verifier_actions: List[Dict]) -> Dict[str, Any]:
        """Compare baseline and verifier action selection patterns."""
        comparison = {
            'total_actions': {
                'baseline': len(baseline_actions),
                'verifier': len(verifier_actions),
                'difference': len(verifier_actions) - len(baseline_actions)
            },
            'action_type_comparison': self._compare_action_types(baseline_actions, verifier_actions),
            'magnitude_comparison': self._compare_action_magnitudes(baseline_actions, verifier_actions),
            'question_type_utilization': self._compare_question_type_utilization(baseline_actions, verifier_actions),
            'baseline_scoring_analysis': self._analyze_baseline_scoring(baseline_actions)
        }
        
        return comparison
    
    def _compare_action_types(self, baseline_actions: List[Dict], verifier_actions: List[Dict]) -> Dict[str, Any]:
        """Compare action type distributions between systems."""
        baseline_types = Counter([a['action_type'] for a in baseline_actions])
        verifier_types = Counter([a['action_type'] for a in verifier_actions])
        
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
    
    def _compare_action_magnitudes(self, baseline_actions: List[Dict], verifier_actions: List[Dict]) -> Dict[str, Any]:
        """Compare action magnitude distributions between systems."""
        baseline_magnitudes = [a['magnitude'] for a in baseline_actions if a['magnitude'] is not None]
        verifier_magnitudes = [a['magnitude'] for a in verifier_actions if a['magnitude'] is not None]
        
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
            'statistical_tests': self._perform_magnitude_statistical_tests(baseline_magnitudes, verifier_magnitudes)
        }
    
    def _perform_magnitude_statistical_tests(self, baseline_mags: List[float], verifier_mags: List[float]) -> Dict[str, Any]:
        """Perform statistical tests on magnitude distributions."""
        if not baseline_mags or not verifier_mags:
            return {'error': 'Insufficient data for statistical tests'}
        
        from scipy import stats
        
        try:
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = stats.mannwhitneyu(baseline_mags, verifier_mags, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_mags) - 1) * np.var(baseline_mags) + 
                                 (len(verifier_mags) - 1) * np.var(verifier_mags)) / 
                                (len(baseline_mags) + len(verifier_mags) - 2))
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
    
    def _compare_question_type_utilization(self, baseline_actions: List[Dict], verifier_actions: List[Dict]) -> Dict[str, Any]:
        """Compare action utilization by question type between systems."""
        baseline_by_type = defaultdict(list)
        verifier_by_type = defaultdict(list)
        
        for action in baseline_actions:
            baseline_by_type[action['question_type']].append(action)
        
        for action in verifier_actions:
            verifier_by_type[action['question_type']].append(action)
        
        comparison = {}
        all_types = set(list(baseline_by_type.keys()) + list(verifier_by_type.keys()))
        
        for qtype in all_types:
            baseline_actions_qtype = baseline_by_type[qtype]
            verifier_actions_qtype = verifier_by_type[qtype]
            
            comparison[qtype] = {
                'baseline_count': len(baseline_actions_qtype),
                'verifier_count': len(verifier_actions_qtype),
                'baseline_avg_magnitude': np.mean([a['magnitude'] for a in baseline_actions_qtype if a['magnitude'] is not None]) if baseline_actions_qtype else 0,
                'verifier_avg_magnitude': np.mean([a['magnitude'] for a in verifier_actions_qtype if a['magnitude'] is not None]) if verifier_actions_qtype else 0,
                'baseline_correct_rate': sum(1 for a in baseline_actions_qtype if a['result'] == 'correct') / len(baseline_actions_qtype) if baseline_actions_qtype else 0,
                'verifier_correct_rate': sum(1 for a in verifier_actions_qtype if a['result'] == 'correct') / len(verifier_actions_qtype) if verifier_actions_qtype else 0
            }
        
        return comparison
    
    def _analyze_baseline_scoring(self, baseline_actions: List[Dict]) -> Dict[str, Any]:
        """Analyze baseline scoring patterns."""
        scored_actions = [a for a in baseline_actions if 'baseline_score' in a]
        
        if not scored_actions:
            return {'error': 'No baseline scores found'}
        
        scores = [a['baseline_score'] for a in scored_actions]
        
        return {
            'score_distribution': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores)
            },
            'score_by_action_type': self._analyze_scores_by_action_type(scored_actions),
            'score_effectiveness_correlation': self._analyze_score_effectiveness_correlation(scored_actions)
        }
    
    def _analyze_scores_by_action_type(self, scored_actions: List[Dict]) -> Dict[str, Any]:
        """Analyze baseline scores by action type."""
        by_type = defaultdict(list)
        
        for action in scored_actions:
            by_type[action['action_type']].append(action['baseline_score'])
        
        result = {}
        for action_type, scores in by_type.items():
            result[action_type] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'count': len(scores)
            }
        
        return result
    
    def _analyze_score_effectiveness_correlation(self, scored_actions: List[Dict]) -> Dict[str, Any]:
        """Analyze correlation between baseline scores and effectiveness."""
        scores = []
        correctness = []
        
        for action in scored_actions:
            scores.append(action['baseline_score'])
            correctness.append(1 if action['result'] == 'correct' else 0)
        
        if len(scores) > 1:
            correlation = np.corrcoef(scores, correctness)[0, 1]
            return {
                'correlation': correlation,
                'sample_size': len(scores),
                'interpretation': 'positive' if correlation > 0.1 else 'negative' if correlation < -0.1 else 'weak'
            }
        
        return {'error': 'Insufficient data for correlation analysis'}
    
    def analyze_world_model_quality(self) -> Dict[str, Any]:
        """
        Analyze world model quality metrics from verifier system.
        
        Returns:
            Dictionary containing world model quality analysis
        """
        if 'verification_metrics' not in self.verifier_data:
            return {'error': 'No verification metrics found in verifier data'}
        
        verification_metrics = self.verifier_data['verification_metrics']
        
        # Extract quality metrics
        quality_data = self._extract_quality_metrics(verification_metrics)
        
        return {
            'overall_quality_stats': self._calculate_overall_quality_stats(quality_data),
            'quality_by_action_type': self._analyze_quality_by_action_type(quality_data),
            'quality_by_question_type': self._analyze_quality_by_question_type(quality_data),
            'quality_effectiveness_correlation': self._analyze_quality_effectiveness_correlation(quality_data),
            'quality_distribution': self._analyze_quality_distribution(quality_data)
        }
    
    def _extract_quality_metrics(self, verification_metrics: Dict) -> List[Dict]:
        """Extract quality metrics from verification data."""
        quality_data = []
        
        for qid, qid_data in verification_metrics.items():
            for step, step_data in qid_data.items():
                for action_type, action_data in step_data.items():
                    # Skip non-dict action_data (e.g., threshold values)
                    if not isinstance(action_data, dict):
                        continue
                    
                    for magnitude, metrics in action_data.items():
                        if isinstance(metrics, dict) and 'consistency_score' in metrics:
                            quality_entry = {
                                'qid': qid,
                                'step': step,
                                'action_type': action_type,
                                'magnitude': magnitude,
                                'consistency_score': metrics.get('consistency_score', 0),
                                'reliability_score': metrics.get('reliability_score', 0),
                                'evidence_quality_score': metrics.get('evidence_quality_score', 0),
                                'helpfulness_score': metrics.get('helpfulness_score', 0),
                                'exploration_score': metrics.get('exploration_score', 0),
                                'claim_acceptance_rate': metrics.get('claim_acceptance_rate', 0),
                                'total_claims': metrics.get('total_claims', 0),
                                'average_confidence': metrics.get('average_confidence', 0)
                            }
                            quality_data.append(quality_entry)
        
        return quality_data
    
    def _calculate_overall_quality_stats(self, quality_data: List[Dict]) -> Dict[str, Any]:
        """Calculate overall quality statistics."""
        if not quality_data:
            return {}
        
        metrics = ['consistency_score', 'reliability_score', 'evidence_quality_score', 
                  'helpfulness_score', 'exploration_score', 'claim_acceptance_rate', 'average_confidence']
        
        stats = {}
        for metric in metrics:
            values = [entry[metric] for entry in quality_data if entry[metric] is not None]
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return stats
    
    def _analyze_quality_by_action_type(self, quality_data: List[Dict]) -> Dict[str, Any]:
        """Analyze quality metrics by action type."""
        action_quality = defaultdict(list)
        
        for entry in quality_data:
            action_type = entry['action_type']
            action_quality[action_type].append(entry)
        
        result = {}
        for action_type, entries in action_quality.items():
            metrics = ['consistency_score', 'reliability_score', 'evidence_quality_score', 
                      'helpfulness_score', 'exploration_score']
            
            action_stats = {}
            for metric in metrics:
                values = [entry[metric] for entry in entries if entry[metric] is not None]
                if values:
                    action_stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
            
            result[action_type] = action_stats
        
        return result
    
    def _analyze_quality_by_question_type(self, quality_data: List[Dict]) -> Dict[str, Any]:
        """Analyze quality metrics by question type."""
        # Map qid to question type
        qid_to_type = {}
        for qid, question_data in self.verifier_questions.items():
            if question_data and 'question' in question_data:
                qid_to_type[qid] = question_data['question'].get('question_type', 'unknown')
        
        question_quality = defaultdict(list)
        
        for entry in quality_data:
            qid = entry['qid']
            question_type = qid_to_type.get(qid, 'unknown')
            question_quality[question_type].append(entry)
        
        result = {}
        for question_type, entries in question_quality.items():
            metrics = ['consistency_score', 'reliability_score', 'evidence_quality_score', 
                      'helpfulness_score', 'exploration_score']
            
            type_stats = {}
            for metric in metrics:
                values = [entry[metric] for entry in entries if entry[metric] is not None]
                if values:
                    type_stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
            
            result[question_type] = type_stats
        
        return result
    
    def _analyze_quality_effectiveness_correlation(self, quality_data: List[Dict]) -> Dict[str, Any]:
        """Analyze correlation between quality metrics and effectiveness."""
        # Map qid to result correctness
        qid_to_result = {}
        for qid, question_data in self.verifier_questions.items():
            if question_data:
                qid_to_result[qid] = question_data.get('result', 'unknown')
        
        # Calculate correlations
        correlations = {}
        metrics = ['consistency_score', 'reliability_score', 'evidence_quality_score', 
                  'helpfulness_score', 'exploration_score']
        
        for metric in metrics:
            values = []
            correctness = []
            
            for entry in quality_data:
                qid = entry['qid']
                result = qid_to_result.get(qid, 'unknown')
                
                if entry[metric] is not None and result in ['correct', 'wrong']:
                    values.append(entry[metric])
                    correctness.append(1 if result == 'correct' else 0)
            
            if len(values) > 1:
                correlation = np.corrcoef(values, correctness)[0, 1]
                correlations[metric] = {
                    'correlation': correlation,
                    'sample_size': len(values)
                }
        
        return correlations
    
    def _analyze_quality_distribution(self, quality_data: List[Dict]) -> Dict[str, Any]:
        """Analyze quality score distributions."""
        metrics = ['consistency_score', 'reliability_score', 'evidence_quality_score', 
                  'helpfulness_score', 'exploration_score']
        
        distributions = {}
        for metric in metrics:
            values = [entry[metric] for entry in quality_data if entry[metric] is not None]
            if values:
                distributions[metric] = {
                    'histogram': np.histogram(values, bins=10, range=(0, 1))[0].tolist(),
                    'percentiles': {
                        '25th': np.percentile(values, 25),
                        '50th': np.percentile(values, 50),
                        '75th': np.percentile(values, 75),
                        '90th': np.percentile(values, 90),
                        '95th': np.percentile(values, 95)
                    }
                }
        
        return distributions
    
    def analyze_world_model_utilization(self) -> Dict[str, Any]:
        """
        Analyze world model utilization metrics.
        
        Returns:
            Dictionary containing utilization analysis
        """
        baseline_utilization = self._analyze_utilization_patterns(self.baseline_questions, is_verifier=False)
        verifier_utilization = self._analyze_utilization_patterns(self.verifier_questions, is_verifier=True)
        
        return {
            'baseline_utilization': baseline_utilization,
            'verifier_utilization': verifier_utilization,
            'utilization_comparison': self._compare_utilization_patterns(baseline_utilization, verifier_utilization),
            'detailed_baseline_verifier_comparison': self._detailed_utilization_comparison()
        }
    
    def _analyze_utilization_patterns(self, questions: Dict[str, Dict], is_verifier: bool) -> Dict[str, Any]:
        """Analyze world model utilization patterns."""
        utilization_stats = {
            'total_questions': len(questions),
            'questions_with_actions': 0,
            'total_actions': 0,
            'actions_per_question': [],
            'action_type_distribution': defaultdict(int),
            'utilization_by_question_type': defaultdict(lambda: {'actions': 0, 'questions': 0})
        }
        
        for qid, question_data in questions.items():
            if not question_data:
                continue
            
            question_type = question_data.get('question', {}).get('question_type', 'unknown')
            prompt_content = question_data.get('prompt', {}).get('content', [])
            
            # Count actions in prompt
            actions = self._parse_actions_from_prompt(prompt_content)
            
            if actions:
                utilization_stats['questions_with_actions'] += 1
                utilization_stats['total_actions'] += len(actions)
                utilization_stats['actions_per_question'].append(len(actions))
                
                for action in actions:
                    utilization_stats['action_type_distribution'][action['type']] += 1
                
                utilization_stats['utilization_by_question_type'][question_type]['actions'] += len(actions)
                utilization_stats['utilization_by_question_type'][question_type]['questions'] += 1
        
        # Calculate averages
        if utilization_stats['actions_per_question']:
            utilization_stats['avg_actions_per_question'] = np.mean(utilization_stats['actions_per_question'])
            utilization_stats['std_actions_per_question'] = np.std(utilization_stats['actions_per_question'])
        else:
            utilization_stats['avg_actions_per_question'] = 0
            utilization_stats['std_actions_per_question'] = 0
        
        # Convert defaultdicts to regular dicts
        utilization_stats['action_type_distribution'] = dict(utilization_stats['action_type_distribution'])
        utilization_stats['utilization_by_question_type'] = dict(utilization_stats['utilization_by_question_type'])
        
        return utilization_stats
    
    def _compare_utilization_patterns(self, baseline_util: Dict, verifier_util: Dict) -> Dict[str, Any]:
        """Compare utilization patterns between systems."""
        comparison = {
            'action_efficiency': {
                'baseline_avg_actions': baseline_util['avg_actions_per_question'],
                'verifier_avg_actions': verifier_util['avg_actions_per_question'],
                'efficiency_ratio': verifier_util['avg_actions_per_question'] / baseline_util['avg_actions_per_question'] if baseline_util['avg_actions_per_question'] > 0 else 0
            },
            'action_diversity': {
                'baseline_unique_actions': len(baseline_util['action_type_distribution']),
                'verifier_unique_actions': len(verifier_util['action_type_distribution']),
                'diversity_improvement': len(verifier_util['action_type_distribution']) - len(baseline_util['action_type_distribution'])
            },
            'utilization_by_question_type': {}
        }
        
        # Compare utilization by question type
        all_question_types = set(list(baseline_util['utilization_by_question_type'].keys()) + 
                                list(verifier_util['utilization_by_question_type'].keys()))
        
        for qtype in all_question_types:
            baseline_data = baseline_util['utilization_by_question_type'].get(qtype, {'actions': 0, 'questions': 0})
            verifier_data = verifier_util['utilization_by_question_type'].get(qtype, {'actions': 0, 'questions': 0})
            
            baseline_avg = baseline_data['actions'] / baseline_data['questions'] if baseline_data['questions'] > 0 else 0
            verifier_avg = verifier_data['actions'] / verifier_data['questions'] if verifier_data['questions'] > 0 else 0
            
            comparison['utilization_by_question_type'][qtype] = {
                'baseline_avg_actions': baseline_avg,
                'verifier_avg_actions': verifier_avg,
                'improvement': verifier_avg - baseline_avg
            }
        
        return comparison
    
    def _detailed_utilization_comparison(self) -> Dict[str, Any]:
        """Provide detailed comparison between baseline and verifier utilization."""
        baseline_actions = self._extract_action_data(self.baseline_questions, is_verifier=False)
        verifier_actions = self._extract_action_data(self.verifier_questions, is_verifier=True)
        
        # Group by question ID for direct comparison
        baseline_by_qid = defaultdict(list)
        verifier_by_qid = defaultdict(list)
        
        for action in baseline_actions:
            baseline_by_qid[action['qid']].append(action)
        
        for action in verifier_actions:
            verifier_by_qid[action['qid']].append(action)
        
        # Direct question-by-question comparison
        comparison_results = {
            'questions_with_actions': {
                'baseline_only': 0,
                'verifier_only': 0,
                'both_systems': 0,
                'neither_system': 0
            },
            'action_count_comparison': [],
            'action_type_differences': defaultdict(int),
            'magnitude_differences': [],
            'effectiveness_improvements': []
        }
        
        all_qids = set(list(baseline_by_qid.keys()) + list(verifier_by_qid.keys()))
        
        for qid in all_qids:
            baseline_q_actions = baseline_by_qid.get(qid, [])
            verifier_q_actions = verifier_by_qid.get(qid, [])
            
            # Count questions by action presence
            has_baseline = len(baseline_q_actions) > 0
            has_verifier = len(verifier_q_actions) > 0
            
            if has_baseline and has_verifier:
                comparison_results['questions_with_actions']['both_systems'] += 1
            elif has_baseline:
                comparison_results['questions_with_actions']['baseline_only'] += 1
            elif has_verifier:
                comparison_results['questions_with_actions']['verifier_only'] += 1
            else:
                comparison_results['questions_with_actions']['neither_system'] += 1
            
            # Compare action counts
            action_count_diff = len(verifier_q_actions) - len(baseline_q_actions)
            comparison_results['action_count_comparison'].append({
                'qid': qid,
                'baseline_count': len(baseline_q_actions),
                'verifier_count': len(verifier_q_actions),
                'difference': action_count_diff
            })
            
            # Compare action types
            baseline_types = set(a['action_type'] for a in baseline_q_actions)
            verifier_types = set(a['action_type'] for a in verifier_q_actions)
            
            for action_type in verifier_types - baseline_types:
                comparison_results['action_type_differences'][f'verifier_only_{action_type}'] += 1
            
            for action_type in baseline_types - verifier_types:
                comparison_results['action_type_differences'][f'baseline_only_{action_type}'] += 1
            
            # Compare magnitudes
            baseline_mags = [a['magnitude'] for a in baseline_q_actions if a['magnitude'] is not None]
            verifier_mags = [a['magnitude'] for a in verifier_q_actions if a['magnitude'] is not None]
            
            if baseline_mags and verifier_mags:
                comparison_results['magnitude_differences'].append({
                    'qid': qid,
                    'baseline_avg': np.mean(baseline_mags),
                    'verifier_avg': np.mean(verifier_mags),
                    'difference': np.mean(verifier_mags) - np.mean(baseline_mags)
                })
            
            # Compare effectiveness
            baseline_correct = sum(1 for a in baseline_q_actions if a['result'] == 'correct')
            verifier_correct = sum(1 for a in verifier_q_actions if a['result'] == 'correct')
            
            if baseline_q_actions and verifier_q_actions:
                baseline_rate = baseline_correct / len(baseline_q_actions)
                verifier_rate = verifier_correct / len(verifier_q_actions)
                
                comparison_results['effectiveness_improvements'].append({
                    'qid': qid,
                    'baseline_rate': baseline_rate,
                    'verifier_rate': verifier_rate,
                    'improvement': verifier_rate - baseline_rate
                })
        
        # Convert defaultdict to regular dict
        comparison_results['action_type_differences'] = dict(comparison_results['action_type_differences'])
        
        # Add summary statistics
        comparison_results['summary'] = {
            'avg_action_count_difference': np.mean([c['difference'] for c in comparison_results['action_count_comparison']]),
            'avg_magnitude_difference': np.mean([m['difference'] for m in comparison_results['magnitude_differences']]) if comparison_results['magnitude_differences'] else 0,
            'avg_effectiveness_improvement': np.mean([e['improvement'] for e in comparison_results['effectiveness_improvements']]) if comparison_results['effectiveness_improvements'] else 0,
            'questions_analyzed': len(all_qids)
        }
        
        return comparison_results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of all world model utilization metrics.
        
        Returns:
            Dictionary containing complete analysis report
        """
        report = {
            'action_selection_quality': self.analyze_action_selection_quality(),
            'world_model_quality': self.analyze_world_model_quality(),
            'world_model_utilization': self.analyze_world_model_utilization(),
            'summary_statistics': self._generate_summary_statistics()
        }
        
        return report
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the analysis."""
        baseline_accuracy = self.baseline_data.get('accuracy', {}).get('all', 0)
        verifier_accuracy = self.verifier_data.get('accuracy', {}).get('all', 0)
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'verifier_accuracy': verifier_accuracy,
            'accuracy_improvement': verifier_accuracy - baseline_accuracy,
            'relative_improvement': (verifier_accuracy - baseline_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0,
            'total_questions_analyzed': len(self.verifier_questions),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def save_report(self, output_path: str) -> None:
        """Save the comprehensive report to a JSON file."""
        report = self.generate_comprehensive_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"World model utilization report saved to: {output_path}")


def main():
    """Main function to run the world model utilization analysis."""
    # Define paths
    baseline_path = "/Users/saurav/Documents/MindJourney/results/svc_test_o4-mini_150_1_2_2_2_spatial_beam_search"
    verifier_path = "/Users/saurav/Documents/MindJourney/results/svc_test_o4-mini_150_1_2_2_2_with_verifier_spatial_beam_search"
    
    # Initialize analyzer
    analyzer = WorldModelUtilizationAnalyzer(baseline_path, verifier_path)
    
    # Generate and save report
    output_file = "/Users/saurav/Documents/MindJourney/utils/further_evals/world_model_utilization_report_reliablity_ablation.json"
    analyzer.save_report(output_file)
    
    # Print summary
    report = analyzer.generate_comprehensive_report()
    summary = report['summary_statistics']
    
    print("\n" + "="*60)
    print("WORLD MODEL UTILIZATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Baseline Accuracy: {summary['baseline_accuracy']:.3f}")
    print(f"Verifier Accuracy: {summary['verifier_accuracy']:.3f}")
    print(f"Accuracy Improvement: {summary['accuracy_improvement']:.3f}")
    print(f"Relative Improvement: {summary['relative_improvement']:.1%}")
    print(f"Questions Analyzed: {summary['total_questions_analyzed']}")
    print("="*60)


if __name__ == "__main__":
    main()
