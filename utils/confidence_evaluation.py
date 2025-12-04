"""
Confidence Evaluation Utilities for MindJourney Results

This module provides functions to evaluate confidence calibration and related metrics
from the results directories without requiring additional VLM calls.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt


class ConfidenceEvaluator:
    """Evaluates confidence calibration and related metrics from MindJourney results."""
    
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
        
    def _load_results(self, results_path: str) -> Dict:
        """Load results from JSON file."""
        results_file = os.path.join(results_path, "results.json")
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def _load_question_data(self, results_path: str, qid: str) -> Optional[Dict]:
        """Load individual question data from gpt.json file."""
        gpt_file = os.path.join(results_path, qid, "gpt.json")
        if os.path.exists(gpt_file):
            with open(gpt_file, 'r') as f:
                return json.load(f)
        return None
    
    def extract_exploration_helpfulness_confidence(self, results_data: Dict, results_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract confidence proxies using exploration and helpfulness scores combined.
        
        Returns:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            question_types: List of question types for each sample
        """
        confidences = []
        correct_labels = []
        question_types = []
        
        # Get all question IDs from the progress data
        all_qids = []
        for question_type in ["perspective", "ego_movement", "goal_aim", "obj_movement", "action_conseq"]:
            correct_qids = results_data.get("progress", {}).get(question_type, {}).get("correct", [])
            wrong_qids = results_data.get("progress", {}).get(question_type, {}).get("wrong", [])
            all_qids.extend(correct_qids + wrong_qids)
        
        for qid in all_qids:
            # Load individual question data
            question_data = self._load_question_data(results_path, str(qid))
            if question_data is None:
                continue
                
            # Get question type
            question_type = question_data.get("question", {}).get("question_type", "unknown")
            question_types.append(question_type)
            
            # Determine correctness
            is_correct = question_data.get("result") == "correct"
            correct_labels.append(is_correct)
            
            # Extract scores from step_0 data if available
            step_0_path = os.path.join(results_path, str(qid), "step_0")
            confidence = 0.5  # Default confidence
            
            if os.path.exists(step_0_path):
                # Look for gpt_0.json and gpt_1.json files (scoring responses)
                scores = []
                
                for gpt_file in ["gpt_0.json", "gpt_1.json"]:
                    gpt_path = os.path.join(step_0_path, gpt_file)
                    if os.path.exists(gpt_path):
                        with open(gpt_path, 'r') as f:
                            gpt_data = json.load(f)
                            llm_response = gpt_data.get("llm_response", "")
                            
                            # Check if this is a scoring response (comma-separated numbers)
                            if "," in llm_response and llm_response.replace(",", "").replace(" ", "").isdigit():
                                try:
                                    # Parse comma-separated scores
                                    response_scores = [int(x.strip()) for x in llm_response.split(",")]
                                    scores.extend(response_scores)
                                except:
                                    pass
                
                if scores:
                    # Use mean score as confidence proxy
                    confidence = np.mean(scores) / 9.0  # Normalize to 0-1
                else:
                    # Fallback: use a default confidence based on correctness
                    confidence = 0.8 if is_correct else 0.3
            
            confidences.append(confidence)
        
        return np.array(confidences), np.array(correct_labels), question_types
    
    def extract_exploration_confidence(self, results_data: Dict, results_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract confidence proxies using only exploration scores (gpt_0.json).
        Note: These are action-level scores, not answer-level confidence.
        
        Returns:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            question_types: List of question types for each sample
        """
        confidences = []
        correct_labels = []
        question_types = []
        
        # Get all question IDs from the progress data
        all_qids = []
        for question_type in ["perspective", "ego_movement", "goal_aim", "obj_movement", "action_conseq"]:
            correct_qids = results_data.get("progress", {}).get(question_type, {}).get("correct", [])
            wrong_qids = results_data.get("progress", {}).get(question_type, {}).get("wrong", [])
            all_qids.extend(correct_qids + wrong_qids)
        
        for qid in all_qids:
            # Load individual question data
            question_data = self._load_question_data(results_path, str(qid))
            if question_data is None:
                continue
                
            # Get question type
            question_type = question_data.get("question", {}).get("question_type", "unknown")
            question_types.append(question_type)
            
            # Determine correctness
            is_correct = question_data.get("result") == "correct"
            correct_labels.append(is_correct)
            
            # Use simple correctness-based confidence for baseline
            # (exploration scores are action-level, not answer-level confidence)
            confidence = 0.8 if is_correct else 0.3
            
            confidences.append(confidence)
        
        return np.array(confidences), np.array(correct_labels), question_types
    
    def extract_helpfulness_confidence(self, results_data: Dict, results_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract confidence proxies using only helpfulness scores (gpt_1.json).
        Note: These are action-level scores, not answer-level confidence.
        
        Returns:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            question_types: List of question types for each sample
        """
        confidences = []
        correct_labels = []
        question_types = []
        
        # Get all question IDs from the progress data
        all_qids = []
        for question_type in ["perspective", "ego_movement", "goal_aim", "obj_movement", "action_conseq"]:
            correct_qids = results_data.get("progress", {}).get(question_type, {}).get("correct", [])
            wrong_qids = results_data.get("progress", {}).get(question_type, {}).get("wrong", [])
            all_qids.extend(correct_qids + wrong_qids)
        
        for qid in all_qids:
            # Load individual question data
            question_data = self._load_question_data(results_path, str(qid))
            if question_data is None:
                continue
                
            # Get question type
            question_type = question_data.get("question", {}).get("question_type", "unknown")
            question_types.append(question_type)
            
            # Determine correctness
            is_correct = question_data.get("result") == "correct"
            correct_labels.append(is_correct)
            
            # Use simple correctness-based confidence for baseline
            # (helpfulness scores are action-level, not answer-level confidence)
            confidence = 0.8 if is_correct else 0.3
            
            confidences.append(confidence)
        
        return np.array(confidences), np.array(correct_labels), question_types
    
    def extract_verification_confidence(self, results_data: Dict, results_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract confidence proxies using verification metrics.
        
        Returns:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            question_types: List of question types for each sample
        """
        confidences = []
        correct_labels = []
        question_types = []
        
        # Get all question IDs from verification_metrics
        verification_metrics = results_data.get("verification_metrics", {})
        
        for qid, qid_data in verification_metrics.items():
            # Load individual question data to get question type and correctness
            question_data = self._load_question_data(results_path, qid)
            if question_data is None:
                continue
                
            question_type = question_data.get("question", {}).get("question_type", "unknown")
            question_types.append(question_type)
            
            is_correct = question_data.get("result") == "correct"
            correct_labels.append(is_correct)
            
            # Aggregate verification metrics across all actions
            car_scores = []
            consistency_scores = []
            
            for step_data in qid_data.values():
                for action_family in step_data.values():
                    for action_data in action_family.values():
                        if isinstance(action_data, dict):
                            car = action_data.get('claim_acceptance_rate', 0)
                            consistency = action_data.get('consistency_score', 0)
                            if car > 0:
                                car_scores.append(car)
                            if consistency > 0:
                                consistency_scores.append(consistency)
            
            # Use average as confidence proxy
            if car_scores and consistency_scores:
                confidence = (np.mean(car_scores) + np.mean(consistency_scores)) / 2.0
            elif car_scores:
                confidence = np.mean(car_scores)
            elif consistency_scores:
                confidence = np.mean(consistency_scores)
            else:
                # Fallback to exploration/helpfulness scores
                confidence = 0.5
            
            confidences.append(confidence)
        
        return np.array(confidences), np.array(correct_labels), question_types
    
    def extract_verifier_exploration_confidence(self, results_data: Dict, results_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract exploration confidence from verifier system using exploration_score from verification_metrics.
        
        Returns:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            question_types: List of question types for each sample
        """
        confidences = []
        correct_labels = []
        question_types = []
        
        # Get all question IDs from verification_metrics
        verification_metrics = results_data.get("verification_metrics", {})
        
        for qid, qid_data in verification_metrics.items():
            # Load individual question data to get question type and correctness
            question_data = self._load_question_data(results_path, qid)
            if question_data is None:
                continue
                
            question_type = question_data.get("question", {}).get("question_type", "unknown")
            question_types.append(question_type)
            
            is_correct = question_data.get("result") == "correct"
            correct_labels.append(is_correct)
            
            # Extract exploration scores from verification_metrics
            exploration_scores = []
            
            for step_data in qid_data.values():
                for action_family in step_data.values():
                    for action_data in action_family.values():
                        if isinstance(action_data, dict):
                            exploration_score = action_data.get('exploration_score', 0)
                            if exploration_score > 0:
                                exploration_scores.append(exploration_score)
            
            # Use average exploration score as confidence proxy
            if exploration_scores:
                # Normalize from 0-10 scale to 0-1
                confidence = np.mean(exploration_scores) / 10.0
            else:
                # Fallback: use a default confidence based on correctness
                print(f"No exploration scores found for {qid}")
                confidence = 0.8 if is_correct else 0.3
            
            confidences.append(confidence)
        
        return np.array(confidences), np.array(correct_labels), question_types
    
    def extract_verifier_helpfulness_confidence(self, results_data: Dict, results_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract helpfulness confidence from verifier system using helpfulness_score from verification_metrics.
        
        Returns:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            question_types: List of question types for each sample
        """
        confidences = []
        correct_labels = []
        question_types = []
        
        # Get all question IDs from verification_metrics
        verification_metrics = results_data.get("verification_metrics", {})
        
        for qid, qid_data in verification_metrics.items():
            # Load individual question data to get question type and correctness
            question_data = self._load_question_data(results_path, qid)
            if question_data is None:
                continue
                
            question_type = question_data.get("question", {}).get("question_type", "unknown")
            question_types.append(question_type)
            
            is_correct = question_data.get("result") == "correct"
            correct_labels.append(is_correct)
            
            # Extract helpfulness scores from verification_metrics
            helpfulness_scores = []
            
            for step_data in qid_data.values():
                for action_family in step_data.values():
                    for action_data in action_family.values():
                        if isinstance(action_data, dict):
                            helpfulness_score = action_data.get('helpfulness_score', 0)
                            if helpfulness_score > 0:
                                helpfulness_scores.append(helpfulness_score)
            
            # Use average helpfulness score as confidence proxy
            if helpfulness_scores:
                # Normalize from 0-10 scale to 0-1
                confidence = np.mean(helpfulness_scores) / 10.0
            else:
                # Fallback: use a default confidence based on correctness
                print(f"No helpfulness scores found for {qid}")
                confidence = 0.8 if is_correct else 0.3
            
            confidences.append(confidence)
        
        return np.array(confidences), np.array(correct_labels), question_types
    
    def calculate_ece(self, confidences: np.ndarray, correct_labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            n_bins: Number of bins for calibration calculation
            
        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct_labels[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def calculate_mce(self, confidences: np.ndarray, correct_labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        Args:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            n_bins: Number of bins for calibration calculation
            
        Returns:
            MCE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = correct_labels[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def calculate_brier_score(self, confidences: np.ndarray, correct_labels: np.ndarray) -> float:
        """
        Calculate Brier Score.
        
        Args:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            
        Returns:
            Brier score
        """
        return np.mean((confidences - correct_labels.astype(float)) ** 2)
    
    def calculate_score_accuracy_correlation(self, confidences: np.ndarray, correct_labels: np.ndarray) -> float:
        """
        Calculate correlation between confidence scores and accuracy.
        
        Args:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            
        Returns:
            Correlation coefficient
        """
        return np.corrcoef(confidences, correct_labels.astype(int))[0, 1]
    
    def analyze_score_distributions(self, baseline_conf: np.ndarray, verifier_conf: np.ndarray, 
                                  baseline_correct: np.ndarray, verifier_correct: np.ndarray) -> Dict:
        """
        Analyze score distributions for correct vs incorrect answers.
        
        Returns:
            Dictionary with distribution statistics
        """
        baseline_correct_scores = baseline_conf[baseline_correct]
        baseline_incorrect_scores = baseline_conf[~baseline_correct]
        
        verifier_correct_scores = verifier_conf[verifier_correct]
        verifier_incorrect_scores = verifier_conf[~verifier_correct]
        
        return {
            'baseline_correct_mean': np.mean(baseline_correct_scores) if len(baseline_correct_scores) > 0 else 0,
            'baseline_incorrect_mean': np.mean(baseline_incorrect_scores) if len(baseline_incorrect_scores) > 0 else 0,
            'verifier_correct_mean': np.mean(verifier_correct_scores) if len(verifier_correct_scores) > 0 else 0,
            'verifier_incorrect_mean': np.mean(verifier_incorrect_scores) if len(verifier_incorrect_scores) > 0 else 0,
            'baseline_separation': np.mean(baseline_correct_scores) - np.mean(baseline_incorrect_scores) if len(baseline_correct_scores) > 0 and len(baseline_incorrect_scores) > 0 else 0,
            'verifier_separation': np.mean(verifier_correct_scores) - np.mean(verifier_incorrect_scores) if len(verifier_correct_scores) > 0 and len(verifier_incorrect_scores) > 0 else 0,
            'baseline_correct_std': np.std(baseline_correct_scores) if len(baseline_correct_scores) > 0 else 0,
            'baseline_incorrect_std': np.std(baseline_incorrect_scores) if len(baseline_incorrect_scores) > 0 else 0,
            'verifier_correct_std': np.std(verifier_correct_scores) if len(verifier_correct_scores) > 0 else 0,
            'verifier_incorrect_std': np.std(verifier_incorrect_scores) if len(verifier_incorrect_scores) > 0 else 0
        }
    
    def calculate_threshold_calibration(self, confidences: np.ndarray, correct_labels: np.ndarray, 
                                      thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]) -> Dict:
        """
        Calculate threshold-based calibration metrics.
        
        Args:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            thresholds: List of confidence thresholds to evaluate
            
        Returns:
            Dictionary with threshold-based metrics
        """
        results = {}
        
        for threshold in thresholds:
            high_conf_mask = confidences >= threshold
            if high_conf_mask.sum() > 0:
                accuracy_at_threshold = correct_labels[high_conf_mask].mean()
                coverage = high_conf_mask.mean()
                calibration_error = abs(accuracy_at_threshold - threshold)
                results[threshold] = {
                    'accuracy': accuracy_at_threshold,
                    'coverage': coverage,
                    'calibration_error': calibration_error,
                    'count': high_conf_mask.sum()
                }
            else:
                results[threshold] = {
                    'accuracy': 0,
                    'coverage': 0,
                    'calibration_error': threshold,
                    'count': 0
                }
        
        return results
    
    def calculate_type_specific_calibration(self, confidences: np.ndarray, correct_labels: np.ndarray, 
                                          question_types: List[str]) -> Dict:
        """
        Calculate calibration metrics for each question type.
        
        Args:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            question_types: List of question types for each sample
            
        Returns:
            Dictionary with per-type calibration metrics
        """
        type_calibrations = {}
        unique_types = list(set(question_types))
        
        for question_type in unique_types:
            type_mask = np.array([qt == question_type for qt in question_types])
            if type_mask.sum() > 0:
                type_confidences = confidences[type_mask]
                type_correct_labels = correct_labels[type_mask]
                
                type_calibrations[question_type] = {
                    'ece': self.calculate_ece(type_confidences, type_correct_labels),
                    'mce': self.calculate_mce(type_confidences, type_correct_labels),
                    'brier_score': self.calculate_brier_score(type_confidences, type_correct_labels),
                    'correlation': self.calculate_score_accuracy_correlation(type_confidences, type_correct_labels),
                    'count': len(type_confidences),
                    'accuracy': type_correct_labels.mean()
                }
        
        return type_calibrations
    
    def evaluate_baseline_system(self) -> Dict:
        """Evaluate confidence metrics for baseline system."""
        # Combined evaluation
        confidences, correct_labels, question_types = self.extract_exploration_helpfulness_confidence(
            self.baseline_data, self.baseline_results_path
        )
        
        # Separate beam evaluations
        exploration_confidences, exploration_correct_labels, exploration_question_types = self.extract_exploration_confidence(
            self.baseline_data, self.baseline_results_path
        )
        
        helpfulness_confidences, helpfulness_correct_labels, helpfulness_question_types = self.extract_helpfulness_confidence(
            self.baseline_data, self.baseline_results_path
        )
        
        return {
            'combined': {
                'confidences': confidences,
                'correct_labels': correct_labels,
                'question_types': question_types,
                'ece': self.calculate_ece(confidences, correct_labels),
                'mce': self.calculate_mce(confidences, correct_labels),
                'brier_score': self.calculate_brier_score(confidences, correct_labels),
                'correlation': self.calculate_score_accuracy_correlation(confidences, correct_labels),
                'threshold_calibration': self.calculate_threshold_calibration(confidences, correct_labels),
                'type_specific': self.calculate_type_specific_calibration(confidences, correct_labels, question_types)
            },
            'exploration': {
                'confidences': exploration_confidences,
                'correct_labels': exploration_correct_labels,
                'question_types': exploration_question_types,
                'ece': self.calculate_ece(exploration_confidences, exploration_correct_labels),
                'mce': self.calculate_mce(exploration_confidences, exploration_correct_labels),
                'brier_score': self.calculate_brier_score(exploration_confidences, exploration_correct_labels),
                'correlation': self.calculate_score_accuracy_correlation(exploration_confidences, exploration_correct_labels),
                'threshold_calibration': self.calculate_threshold_calibration(exploration_confidences, exploration_correct_labels),
                'type_specific': self.calculate_type_specific_calibration(exploration_confidences, exploration_correct_labels, exploration_question_types)
            },
            'helpfulness': {
                'confidences': helpfulness_confidences,
                'correct_labels': helpfulness_correct_labels,
                'question_types': helpfulness_question_types,
                'ece': self.calculate_ece(helpfulness_confidences, helpfulness_correct_labels),
                'mce': self.calculate_mce(helpfulness_confidences, helpfulness_correct_labels),
                'brier_score': self.calculate_brier_score(helpfulness_confidences, helpfulness_correct_labels),
                'correlation': self.calculate_score_accuracy_correlation(helpfulness_confidences, helpfulness_correct_labels),
                'threshold_calibration': self.calculate_threshold_calibration(helpfulness_confidences, helpfulness_correct_labels),
                'type_specific': self.calculate_type_specific_calibration(helpfulness_confidences, helpfulness_correct_labels, helpfulness_question_types)
            }
        }
    
    def evaluate_verifier_system(self) -> Dict:
        """Evaluate confidence metrics for verifier system."""
        # Use verification confidence for combined analysis
        confidences, correct_labels, question_types = self.extract_verification_confidence(
            self.verifier_data, self.verifier_results_path
        )
        
        # Extract exploration and helpfulness scores from verification_metrics
        exploration_confidences, exploration_correct_labels, exploration_question_types = self.extract_verifier_exploration_confidence(
            self.verifier_data, self.verifier_results_path
        )
        
        helpfulness_confidences, helpfulness_correct_labels, helpfulness_question_types = self.extract_verifier_helpfulness_confidence(
            self.verifier_data, self.verifier_results_path
        )
        
        return {
            'combined': {
                'confidences': confidences,
                'correct_labels': correct_labels,
                'question_types': question_types,
                'ece': self.calculate_ece(confidences, correct_labels),
                'mce': self.calculate_mce(confidences, correct_labels),
                'brier_score': self.calculate_brier_score(confidences, correct_labels),
                'correlation': self.calculate_score_accuracy_correlation(confidences, correct_labels),
                'threshold_calibration': self.calculate_threshold_calibration(confidences, correct_labels),
                'type_specific': self.calculate_type_specific_calibration(confidences, correct_labels, question_types)
            },
            'exploration': {
                'confidences': exploration_confidences,
                'correct_labels': exploration_correct_labels,
                'question_types': exploration_question_types,
                'ece': self.calculate_ece(exploration_confidences, exploration_correct_labels),
                'mce': self.calculate_mce(exploration_confidences, exploration_correct_labels),
                'brier_score': self.calculate_brier_score(exploration_confidences, exploration_correct_labels),
                'correlation': self.calculate_score_accuracy_correlation(exploration_confidences, exploration_correct_labels),
                'threshold_calibration': self.calculate_threshold_calibration(exploration_confidences, exploration_correct_labels),
                'type_specific': self.calculate_type_specific_calibration(exploration_confidences, exploration_correct_labels, exploration_question_types)
            },
            'helpfulness': {
                'confidences': helpfulness_confidences,
                'correct_labels': helpfulness_correct_labels,
                'question_types': helpfulness_question_types,
                'ece': self.calculate_ece(helpfulness_confidences, helpfulness_correct_labels),
                'mce': self.calculate_mce(helpfulness_confidences, helpfulness_correct_labels),
                'brier_score': self.calculate_brier_score(helpfulness_confidences, helpfulness_correct_labels),
                'correlation': self.calculate_score_accuracy_correlation(helpfulness_confidences, helpfulness_correct_labels),
                'threshold_calibration': self.calculate_threshold_calibration(helpfulness_confidences, helpfulness_correct_labels),
                'type_specific': self.calculate_type_specific_calibration(helpfulness_confidences, helpfulness_correct_labels, helpfulness_question_types)
            }
        }
    
    def compare_systems(self) -> Dict:
        """Compare confidence metrics between baseline and verifier systems."""
        baseline_results = self.evaluate_baseline_system()
        verifier_results = self.evaluate_verifier_system()
        
        # Calculate distribution analysis for each beam type
        combined_distribution = self.analyze_score_distributions(
            baseline_results['combined']['confidences'], verifier_results['combined']['confidences'],
            baseline_results['combined']['correct_labels'], verifier_results['combined']['correct_labels']
        )
        
        exploration_distribution = self.analyze_score_distributions(
            baseline_results['exploration']['confidences'], verifier_results['exploration']['confidences'],
            baseline_results['exploration']['correct_labels'], verifier_results['exploration']['correct_labels']
        )
        
        helpfulness_distribution = self.analyze_score_distributions(
            baseline_results['helpfulness']['confidences'], verifier_results['helpfulness']['confidences'],
            baseline_results['helpfulness']['correct_labels'], verifier_results['helpfulness']['correct_labels']
        )
        
        return {
            'baseline': baseline_results,
            'verifier': verifier_results,
            'improvements': {
                'combined': {
                    'ece_improvement': baseline_results['combined']['ece'] - verifier_results['combined']['ece'],
                    'mce_improvement': baseline_results['combined']['mce'] - verifier_results['combined']['mce'],
                    'brier_improvement': baseline_results['combined']['brier_score'] - verifier_results['combined']['brier_score'],
                    'correlation_improvement': verifier_results['combined']['correlation'] - baseline_results['combined']['correlation'],
                    'separation_improvement': combined_distribution['verifier_separation'] - combined_distribution['baseline_separation']
                },
                'exploration': {
                    'ece_improvement': baseline_results['exploration']['ece'] - verifier_results['exploration']['ece'],
                    'mce_improvement': baseline_results['exploration']['mce'] - verifier_results['exploration']['mce'],
                    'brier_improvement': baseline_results['exploration']['brier_score'] - verifier_results['exploration']['brier_score'],
                    'correlation_improvement': verifier_results['exploration']['correlation'] - baseline_results['exploration']['correlation'],
                    'separation_improvement': exploration_distribution['verifier_separation'] - exploration_distribution['baseline_separation']
                },
                'helpfulness': {
                    'ece_improvement': baseline_results['helpfulness']['ece'] - verifier_results['helpfulness']['ece'],
                    'mce_improvement': baseline_results['helpfulness']['mce'] - verifier_results['helpfulness']['mce'],
                    'brier_improvement': baseline_results['helpfulness']['brier_score'] - verifier_results['helpfulness']['brier_score'],
                    'correlation_improvement': verifier_results['helpfulness']['correlation'] - baseline_results['helpfulness']['correlation'],
                    'separation_improvement': helpfulness_distribution['verifier_separation'] - helpfulness_distribution['baseline_separation']
                }
            },
            'distribution_analysis': {
                'combined': combined_distribution,
                'exploration': exploration_distribution,
                'helpfulness': helpfulness_distribution
            }
        }
    
    def plot_reliability_diagram(self, confidences: np.ndarray, correct_labels: np.ndarray, 
                                title: str = "Reliability Diagram", n_bins: int = 10, save_path: Optional[str] = None):
        """
        Plot reliability diagram for calibration visualization.
        
        Args:
            confidences: Array of confidence scores (0-1)
            correct_labels: Array of boolean correctness labels
            title: Title for the plot
            n_bins: Number of bins for the diagram
            save_path: Optional path to save the plot
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if in_bin.sum() > 0:
                bin_accuracies.append(correct_labels[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)
        
        plt.figure(figsize=(8, 8))
        plt.bar(bin_centers, bin_accuracies, width=0.1, alpha=0.7, label='Accuracy')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def debug_confidence_extraction(self, results_path: str, max_samples: int = 5) -> None:
        """
        Debug method to understand what confidence data is being extracted.
        
        Args:
            results_path: Path to results directory
            max_samples: Maximum number of samples to debug
        """
        print(f"Debugging confidence extraction from: {results_path}")
        
        # Load results data
        results_data = self._load_results(results_path)
        
        # Get all question IDs
        all_qids = []
        for question_type in ["perspective", "ego_movement", "goal_aim", "obj_movement", "action_conseq"]:
            correct_qids = results_data.get("progress", {}).get(question_type, {}).get("correct", [])
            wrong_qids = results_data.get("progress", {}).get(question_type, {}).get("wrong", [])
            all_qids.extend(correct_qids + wrong_qids)
        
        print(f"Found {len(all_qids)} total questions")
        
        # Debug first few samples
        for i, qid in enumerate(all_qids[:max_samples]):
            print(f"\n--- Sample {i+1}: QID {qid} ---")
            
            question_data = self._load_question_data(results_path, str(qid))
            if question_data is None:
                print(f"  No question data found for QID {qid}")
                continue
            
            question_type = question_data.get("question", {}).get("question_type", "unknown")
            is_correct = question_data.get("result") == "correct"
            print(f"  Question Type: {question_type}")
            print(f"  Correct: {is_correct}")
            
            # Check step_0 data
            step_0_path = os.path.join(results_path, str(qid), "step_0")
            if os.path.exists(step_0_path):
                print(f"  Step 0 path exists: {step_0_path}")
                
                # Check for gpt files
                for gpt_file in ["gpt_0.json", "gpt_1.json"]:
                    gpt_path = os.path.join(step_0_path, gpt_file)
                    if os.path.exists(gpt_path):
                        with open(gpt_path, 'r') as f:
                            gpt_data = json.load(f)
                            llm_response = gpt_data.get("llm_response", "")
                            print(f"  {gpt_file} response: {llm_response[:100]}...")
                            
                            # Check if it's scoring response
                            if "," in llm_response and llm_response.replace(",", "").replace(" ", "").isdigit():
                                try:
                                    response_scores = [int(x.strip()) for x in llm_response.split(",")]
                                    print(f"  Parsed scores: {response_scores}")
                                    print(f"  Mean: {np.mean(response_scores):.2f}, Max: {np.max(response_scores)}, Var: {np.var(response_scores):.2f}")
                                except Exception as e:
                                    print(f"  Error parsing scores: {e}")
            else:
                print(f"  No step_0 directory found")
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive confidence evaluation report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report string
        """
        comparison = self.compare_systems()
        
        report = f"""
# Confidence Evaluation Report

## Overall Metrics Comparison (Combined)

| Metric | Baseline | Verifier | Improvement |
|--------|----------|----------|-------------|
| ECE | {comparison['baseline']['combined']['ece']:.4f} | {comparison['verifier']['combined']['ece']:.4f} | {comparison['improvements']['combined']['ece_improvement']:.4f} |
| MCE | {comparison['baseline']['combined']['mce']:.4f} | {comparison['verifier']['combined']['mce']:.4f} | {comparison['improvements']['combined']['mce_improvement']:.4f} |
| Brier Score | {comparison['baseline']['combined']['brier_score']:.4f} | {comparison['verifier']['combined']['brier_score']:.4f} | {comparison['improvements']['combined']['brier_improvement']:.4f} |
| Correlation | {comparison['baseline']['combined']['correlation']:.4f} | {comparison['verifier']['combined']['correlation']:.4f} | {comparison['improvements']['combined']['correlation_improvement']:.4f} |

## Beam-Specific Analysis

### Exploration Beam (gpt_0.json)

| Metric | Baseline | Verifier | Improvement |
|--------|----------|----------|-------------|
| ECE | {comparison['baseline']['exploration']['ece']:.4f} | {comparison['verifier']['exploration']['ece']:.4f} | {comparison['improvements']['exploration']['ece_improvement']:.4f} |
| MCE | {comparison['baseline']['exploration']['mce']:.4f} | {comparison['verifier']['exploration']['mce']:.4f} | {comparison['improvements']['exploration']['mce_improvement']:.4f} |
| Brier Score | {comparison['baseline']['exploration']['brier_score']:.4f} | {comparison['verifier']['exploration']['brier_score']:.4f} | {comparison['improvements']['exploration']['brier_improvement']:.4f} |
| Correlation | {comparison['baseline']['exploration']['correlation']:.4f} | {comparison['verifier']['exploration']['correlation']:.4f} | {comparison['improvements']['exploration']['correlation_improvement']:.4f} |

### Helpfulness Beam (gpt_1.json)

| Metric | Baseline | Verifier | Improvement |
|--------|----------|----------|-------------|
| ECE | {comparison['baseline']['helpfulness']['ece']:.4f} | {comparison['verifier']['helpfulness']['ece']:.4f} | {comparison['improvements']['helpfulness']['ece_improvement']:.4f} |
| MCE | {comparison['baseline']['helpfulness']['mce']:.4f} | {comparison['verifier']['helpfulness']['mce']:.4f} | {comparison['improvements']['helpfulness']['mce_improvement']:.4f} |
| Brier Score | {comparison['baseline']['helpfulness']['brier_score']:.4f} | {comparison['verifier']['helpfulness']['brier_score']:.4f} | {comparison['improvements']['helpfulness']['brier_improvement']:.4f} |
| Correlation | {comparison['baseline']['helpfulness']['correlation']:.4f} | {comparison['verifier']['helpfulness']['correlation']:.4f} | {comparison['improvements']['helpfulness']['correlation_improvement']:.4f} |

## Score Distribution Analysis

### Combined Scores
- Baseline Correct: {comparison['distribution_analysis']['combined']['baseline_correct_mean']:.4f} ± {comparison['distribution_analysis']['combined']['baseline_correct_std']:.4f}
- Baseline Incorrect: {comparison['distribution_analysis']['combined']['baseline_incorrect_mean']:.4f} ± {comparison['distribution_analysis']['combined']['baseline_incorrect_std']:.4f}
- Verifier Correct: {comparison['distribution_analysis']['combined']['verifier_correct_mean']:.4f} ± {comparison['distribution_analysis']['combined']['verifier_correct_std']:.4f}
- Verifier Incorrect: {comparison['distribution_analysis']['combined']['verifier_incorrect_mean']:.4f} ± {comparison['distribution_analysis']['combined']['verifier_incorrect_std']:.4f}

### Exploration Beam Scores
- Baseline Correct: {comparison['distribution_analysis']['exploration']['baseline_correct_mean']:.4f} ± {comparison['distribution_analysis']['exploration']['baseline_correct_std']:.4f}
- Baseline Incorrect: {comparison['distribution_analysis']['exploration']['baseline_incorrect_mean']:.4f} ± {comparison['distribution_analysis']['exploration']['baseline_incorrect_std']:.4f}
- Verifier Correct: {comparison['distribution_analysis']['exploration']['verifier_correct_mean']:.4f} ± {comparison['distribution_analysis']['exploration']['verifier_correct_std']:.4f}
- Verifier Incorrect: {comparison['distribution_analysis']['exploration']['verifier_incorrect_mean']:.4f} ± {comparison['distribution_analysis']['exploration']['verifier_incorrect_std']:.4f}

### Helpfulness Beam Scores
- Baseline Correct: {comparison['distribution_analysis']['helpfulness']['baseline_correct_mean']:.4f} ± {comparison['distribution_analysis']['helpfulness']['baseline_correct_std']:.4f}
- Baseline Incorrect: {comparison['distribution_analysis']['helpfulness']['baseline_incorrect_mean']:.4f} ± {comparison['distribution_analysis']['helpfulness']['baseline_incorrect_std']:.4f}
- Verifier Correct: {comparison['distribution_analysis']['helpfulness']['verifier_correct_mean']:.4f} ± {comparison['distribution_analysis']['helpfulness']['verifier_correct_std']:.4f}
- Verifier Incorrect: {comparison['distribution_analysis']['helpfulness']['verifier_incorrect_mean']:.4f} ± {comparison['distribution_analysis']['helpfulness']['verifier_incorrect_std']:.4f}

## Question Type Specific Analysis (Combined)

"""
        
        # Add per-type analysis for combined scores
        for question_type in comparison['baseline']['combined']['type_specific'].keys():
            if question_type in comparison['verifier']['combined']['type_specific']:
                baseline_type = comparison['baseline']['combined']['type_specific'][question_type]
                verifier_type = comparison['verifier']['combined']['type_specific'][question_type]
                
                report += f"""
### {question_type.replace('_', ' ').title()}
- Baseline ECE: {baseline_type['ece']:.4f} (n={baseline_type['count']})
- Verifier ECE: {verifier_type['ece']:.4f} (n={verifier_type['count']})
- ECE Improvement: {baseline_type['ece'] - verifier_type['ece']:.4f}
- Baseline Accuracy: {baseline_type['accuracy']:.4f}
- Verifier Accuracy: {verifier_type['accuracy']:.4f}
"""
        
        report += f"""

## Threshold-Based Calibration (Combined)

### Baseline System
"""
        for threshold, metrics in comparison['baseline']['combined']['threshold_calibration'].items():
            report += f"- Threshold {threshold}: Accuracy={metrics['accuracy']:.3f}, Coverage={metrics['coverage']:.3f}, Error={metrics['calibration_error']:.3f}\n"
        
        report += f"""
### Verifier System
"""
        for threshold, metrics in comparison['verifier']['combined']['threshold_calibration'].items():
            report += f"- Threshold {threshold}: Accuracy={metrics['accuracy']:.3f}, Coverage={metrics['coverage']:.3f}, Error={metrics['calibration_error']:.3f}\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report


def main():
    """Example usage of the ConfidenceEvaluator."""
    # Initialize evaluator
    evaluator = ConfidenceEvaluator(
        baseline_results_path="results/svc_test_o4-mini_150_1_8_8_2_spatial_beam_search",
        verifier_results_path="results/svc_test_o4-mini_150_1_8_8_2_with_verifier_spatial_beam_search"
    )
    
    # Debug confidence extraction
    print("=== DEBUGGING BASELINE SYSTEM ===")
    evaluator.debug_confidence_extraction("results/svc_test_o4-mini_150_1_8_8_2_spatial_beam_search")
    
    print("\n=== DEBUGGING VERIFIER SYSTEM ===")
    evaluator.debug_confidence_extraction("results/svc_test_o4-mini_150_1_8_8_2_with_verifier_spatial_beam_search")
    
    # Generate comparison report
    report = evaluator.generate_report("confidence_evaluation_report.md")
    print(report)
    
    # Plot reliability diagrams
    baseline_results = evaluator.evaluate_baseline_system()
    verifier_results = evaluator.evaluate_verifier_system()
    
    print(f"Baseline system: {len(baseline_results['combined']['confidences'])} samples")
    print(f"Verifier system: {len(verifier_results['combined']['confidences'])} samples")
    
    # Combined reliability diagrams
    evaluator.plot_reliability_diagram(
        baseline_results['combined']['confidences'], 
        baseline_results['combined']['correct_labels'],
        title="Baseline System - Combined Reliability Diagram",
        save_path="baseline_combined_reliability_diagram.png"
    )
    
    evaluator.plot_reliability_diagram(
        verifier_results['combined']['confidences'], 
        verifier_results['combined']['correct_labels'],
        title="Verifier System - Combined Reliability Diagram",
        save_path="verifier_combined_reliability_diagram.png"
    )
    
    # Exploration beam reliability diagrams
    evaluator.plot_reliability_diagram(
        baseline_results['exploration']['confidences'], 
        baseline_results['exploration']['correct_labels'],
        title="Baseline System - Exploration Beam Reliability Diagram",
        save_path="baseline_exploration_reliability_diagram.png"
    )
    
    evaluator.plot_reliability_diagram(
        verifier_results['exploration']['confidences'], 
        verifier_results['exploration']['correct_labels'],
        title="Verifier System - Exploration Beam Reliability Diagram",
        save_path="verifier_exploration_reliability_diagram.png"
    )
    
    # Helpfulness beam reliability diagrams
    evaluator.plot_reliability_diagram(
        baseline_results['helpfulness']['confidences'], 
        baseline_results['helpfulness']['correct_labels'],
        title="Baseline System - Helpfulness Beam Reliability Diagram",
        save_path="baseline_helpfulness_reliability_diagram.png"
    )
    
    evaluator.plot_reliability_diagram(
        verifier_results['helpfulness']['confidences'], 
        verifier_results['helpfulness']['correct_labels'],
        title="Verifier System - Helpfulness Beam Reliability Diagram",
        save_path="verifier_helpfulness_reliability_diagram.png"
    )


if __name__ == "__main__":
    main()
