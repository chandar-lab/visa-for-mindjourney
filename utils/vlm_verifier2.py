"""
VLM Verifier for Video Chain-of-Thought (Video COT) approach.
Implements zero-training verification using off-the-shelf VLMs.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.vlm_wrapper import VLMWrapper
from utils.prompt_formatting import (
    format_image_comparison_prompt,
    format_image_comparison_prompt_v2,
    format_verification_prompt_claim_generation,
    format_verification_prompt_claim_verification,
    format_verification_prompt_claim_verification_with_probs
)


class MicroClaimGenerator:
    """Generates micro-claims from video frames and action descriptions."""
    
    def __init__(self, vlm: VLMWrapper):
        self.vlm = vlm

    def _generate_claims_from_images(self, 
                                action_description: str, 
                                images: List[str], 
                                question: str,
                                answer_choices: Optional[List[str]] = None,
                                question_type: Optional[str] = None) -> List[Dict]:
        """
        Generate micro-claims from two images (before and after action).
        
        Args:
            action_description: Description of the action taken
            images: List of two image paths [before, after]
            question: The original spatial reasoning question
            answer_choices: Optional list of answer choices to focus claims on
            
        Returns:
            List of micro-claim dictionaries
        """
        try:
            # Use MMSI-optimized prompt for image comparison
            sys_prompt, content = format_image_comparison_prompt_v2(
                action_description=action_description,
                images=images,
                question=question,
                answer_choices=answer_choices,
                question_type=question_type
            )
            
            response = self.vlm.run_prompt("claim_generation", sys_prompt, content)
            claims = self._parse_claims(response)
            return claims
        except Exception as e:
            print(f"Error generating claims from images: {e}")
            return []
    
    def generate_claims(self, 
                       action_description: str, 
                       video_frames: List[str], 
                       question: str,
                       frame_indices: List[int],
                       answer_choices: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate micro-claims for a specific action and frame sequence.
        
        Args:
            action_description: Description of the action taken (e.g., "turn left 25 degrees")
            video_frames: List of paths to video frame images
            question: The original spatial reasoning question
            frame_indices: List of frame indices to analyze
            answer_choices: Optional list of answer choices to focus claims on
            
        Returns:
            List of micro-claim dictionaries with claim text and frame references
        """
        claims = []
        
        # Group frames into logical segments for claim generation
        # Input: [0, 4, 8, 12] (frames 0, 4, 8, 12)
        # Output: [
        #   {'indices': [0], 'start': 0, 'end': 0},
        #   {'indices': [4], 'start': 4, 'end': 4}, 
        #   {'indices': [8], 'start': 8, 'end': 8},
        #   {'indices': [12], 'start': 12, 'end': 12}
        # ]

        # Input: [0, 1, 2, 8, 9, 10] (consecutive groups)
        # Output: [
        #   {'indices': [0, 1, 2], 'start': 0, 'end': 2},
        #   {'indices': [8, 9, 10], 'start': 8, 'end': 10}
        # ]
        frame_segments = self._segment_frames(frame_indices)
        
        for segment in frame_segments:
            # Map segment indices to video_frames list positions
            segment_frames = []
            for idx in segment['indices']:
                if idx in frame_indices:
                    frame_pos = frame_indices.index(idx)
                    if frame_pos < len(video_frames):
                        segment_frames.append(video_frames[frame_pos])
            frame_range = f"frames {segment['start']}-{segment['end']}"
            
            # Generate claims for this segment
            segment_claims = self._generate_segment_claims(
                action_description, segment_frames, frame_range, segment['indices'], question, answer_choices
            )
            claims.extend(segment_claims)
        
        return claims
    
    def _segment_frames(self, frame_indices: List[int], CLOSE_FRAMES_THRESHOLD: int = 18) -> List[Dict]:
        """Segment frame indices into logical groups for claim generation."""
        if len(frame_indices) <= 3:
            return [{'indices': frame_indices, 'start': frame_indices[0], 'end': frame_indices[-1]}]
        
        # Group consecutive frames
        segments = []
        current_segment = [frame_indices[0]]
        
        for i in range(1, len(frame_indices)):
            if frame_indices[i] - frame_indices[i-1] <= CLOSE_FRAMES_THRESHOLD:  # Consecutive or close frames
                current_segment.append(frame_indices[i])
            else:
                segments.append({
                    'indices': current_segment,
                    'start': current_segment[0],
                    'end': current_segment[-1]
                })
                current_segment = [frame_indices[i]]
        
        segments.append({
            'indices': current_segment,
            'start': current_segment[0],
            'end': current_segment[-1]
        })
        
        return segments
    
    def _generate_segment_claims(self, 
                                action_description: str, 
                                frames: List[str], 
                                frame_range: str,
                                frame_indices: List[int],
                                question: str,
                                answer_choices: Optional[List[str]] = None) -> List[Dict]:
        """Generate micro-claims for a specific frame segment."""
        
        try:
            # Use the proper prompt formatting function
            sys_prompt, content = format_verification_prompt_claim_generation(
                action_description=action_description,
                frames=frames,
                frame_range=frame_range,
                frame_indices=frame_indices,
                question=question
            )
            
            response = self.vlm.run_prompt("claim_generation", sys_prompt, content)
            claims = self._parse_claims(response, frame_range)
            return claims
        except Exception as e:
            print(f"Error generating claims: {e}")
            return []
    
    def _parse_claims(self, response, frame_range=None) -> List[Dict]:
        """Parse micro-claims from VLM response."""
        claims = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                claim_text = line[2:].strip()
                if claim_text:
                    claims.append({
                        'text': claim_text,
                        'frame_range': 'before and after action' if frame_range is None else frame_range,
                        'type': self._classify_claim_type(claim_text)
                    })
        
        return claims
    
    def _classify_claim_type(self, claim_text: str) -> str:
        """Classify the type of claim for analysis purposes."""
        claim_lower = claim_text.lower()
        
        if any(word in claim_lower for word in ['left', 'right', 'behind', 'in front', 'beside', 'next to']):
            return 'spatial_relationship'
        elif any(word in claim_lower for word in ['red', 'blue', 'green', 'color', 'shape', 'size', 'material']):
            return 'object_property'
        elif any(word in claim_lower for word in ['appears', 'disappears', 'moves', 'changes', 'becomes']):
            return 'dynamic_change'
        else:
            return 'general_observation'


class ClaimVerifier:
    """Verifies micro-claims against video frames."""
    
    def __init__(self, vlm: VLMWrapper):
        self.vlm = vlm
    
    def verify_claim(self, claim: Dict, frames: List[str], reason: bool = True) -> Dict:
        """
        Verify a single micro-claim against the provided frames.
        
        Args:
            claim: Micro-claim dictionary with 'text' and 'frame_range'
            frames: List of frame paths to verify against
            reason: Whether to include reasoning in the verification prompt (default: True)
            
        Returns:
            Verification result dictionary
        """
        try:
            # Use the proper prompt formatting function
            sys_prompt, content = format_verification_prompt_claim_verification(
                claim=claim,
                frames=frames,
                reason=reason
            )
            
            response = self.vlm.run_prompt("claim_verification", sys_prompt, content)
            return self._parse_verification(response, claim)
        except Exception as e:
            print(f"Error verifying claim: {e}")
            return {
                'verdict': 'INSUFFICIENT',
                'confidence': 0.0,
                'reasoning': f"Error during verification: {str(e)}",
                # 'probabilities': {'ENTAILED': 0.33, 'CONTRADICTED': 0.33, 'INSUFFICIENT': 0.33},
            }

    def _parse_verification_with_probs(self, response: str, claim: Dict) -> Dict:
        """Parse verification response with single-letter verdicts and probability distributions."""
        lines = response.strip().split('\n')
        verdict = 'I'  # Default to INSUFFICIENT
        reasoning = "No reasoning provided"
        probabilities = {'ENTAILED': 0.33, 'CONTRADICTED': 0.33, 'INSUFFICIENT': 0.33}
        
        for line in lines:
            line = line.strip()
            
            # Parse verdict (single letter)
            if line.startswith('VERDICT:'):
                verdict_letter = line[8:].strip().upper()
                if verdict_letter in ['E', 'C', 'I']:
                    verdict = verdict_letter
                else:
                    print(f"Warning: Invalid verdict '{verdict_letter}', using default 'I'")
                    verdict = 'I'
            
            # Parse probabilities
            elif line.startswith('p(E):'):
                try:
                    probabilities['ENTAILED'] = float(line[5:].strip())
                except ValueError:
                    print(f"Warning: Could not parse p(E) '{line[5:].strip()}', using default")
                    probabilities['ENTAILED'] = 0.33
                    
            elif line.startswith('p(C):'):
                try:
                    probabilities['CONTRADICTED'] = float(line[5:].strip())
                except ValueError:
                    print(f"Warning: Could not parse p(C) '{line[5:].strip()}', using default")
                    probabilities['CONTRADICTED'] = 0.33
                    
            elif line.startswith('p(I):'):
                try:
                    probabilities['INSUFFICIENT'] = float(line[5:].strip())
                except ValueError:
                    print(f"Warning: Could not parse p(I) '{line[5:].strip()}', using default")
                    probabilities['INSUFFICIENT'] = 0.33
            
            # Parse reasoning
            elif line.startswith('REASONING:'):
                reasoning = line[10:].strip()
        
        # Normalize probabilities to ensure they sum to 1.0
        total = sum(probabilities.values())
        if total > 0:
            for key in probabilities:
                probabilities[key] /= total
        else:
            # Fallback if all probabilities are 0
            probabilities = {'ENTAILED': 0.33, 'CONTRADICTED': 0.33, 'INSUFFICIENT': 0.34}
        
        # Validate that verdict matches highest probability
        max_prob_key = max(probabilities, key=probabilities.get)
        verdict_map = {'ENTAILED': 'E', 'CONTRADICTED': 'C', 'INSUFFICIENT': 'I'}
        expected_verdict = verdict_map[max_prob_key]
        
        if verdict != expected_verdict:
            print(f"Warning: Verdict '{verdict}' doesn't match highest probability '{expected_verdict}'")
            # Optionally correct the verdict
            verdict = expected_verdict
            
        # Compute confidence as the probability of the chosen verdict
        confidence = probabilities[{'E': 'ENTAILED', 'C': 'CONTRADICTED', 'I': 'INSUFFICIENT'}[verdict]]
        
        # Convert single letter back to full verdict name for compatibility
        verdict_map_to_full = {'E': 'ENTAILED', 'C': 'CONTRADICTED', 'I': 'INSUFFICIENT'}
        verdict = verdict_map_to_full[verdict_letter]
        return {
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning,
            'probabilities': probabilities,
            'claim': claim
        }
    
    def _parse_verification(self, response: str, claim: Dict) -> Dict:
        """Parse verification response from VLM with new semantic format and confidence scoring."""
        lines = response.strip().split('\n')
        verdict = 'INSUFFICIENT'  # Default fallback
        reasoning = "No reasoning provided"
        confidence = 0.5  # Default fallback
        
        for line in lines:
            line = line.strip()
            if line.startswith('VERDICT:'):
                verdict_text = line[8:].strip().upper()
                if verdict_text in ['ENTAILED', 'CONTRADICTED', 'INSUFFICIENT']:
                    verdict = verdict_text
                else:
                    # Handle legacy format if still present
                    if verdict_text in ['ACCEPT', 'REJECT']:
                        verdict = 'ENTAILED' if verdict_text == 'ACCEPT' else 'CONTRADICTED'
                    elif verdict_text == 'UNCERTAIN':
                        verdict = 'INSUFFICIENT'
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence_text = line[11:].strip()
                    confidence = float(confidence_text)
                    # Clamp confidence to valid range
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    print(f"Warning: Could not parse confidence '{confidence_text}', using fallback")
                    confidence = 0.5
            elif line.startswith('REASONING:'):
                reasoning = line[10:].strip()
        
        # Fallback confidence estimation if not provided or parsing failed
        if confidence == 0.5 and 'confidence' not in response.lower():
            confidence = self._estimate_confidence_from_reasoning(verdict, reasoning)

        # confidence = self.calculate_weighted_score(verdict, confidence)
        return {
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning,
            'claim': claim
        }

    def calculate_weighted_score(self,verdict, confidence):
        """
            High confidence + ENTAILED: Strong candidates for exploration
            High confidence + CONTRADICTED: Good contrastive evidence
            Low confidence + any verdict: Prune or require additional verification
        """
        base_scores = {
            "ENTAILED": 1.0,
            "CONTRADICTED": 0.0, 
            "INSUFFICIENT": 0.3
        }
        return base_scores[verdict] * confidence


    def _estimate_confidence_from_reasoning(self, verdict: str, reasoning: str) -> float:
        """Fallback method to estimate confidence from reasoning text when explicit confidence is not provided."""
        reasoning_lower = reasoning.lower()
        
        # High confidence indicators
        high_conf_indicators = ['clearly', 'obviously', 'definitely', 'unambiguously', 'perfectly clear', 'crystal clear']
        # Medium confidence indicators  
        med_conf_indicators = ['appears', 'seems', 'likely', 'probably', 'mostly', 'generally']
        # Low confidence indicators
        low_conf_indicators = ['unclear', 'ambiguous', 'uncertain', 'difficult to', 'hard to', 'not sure', 'unclear']
        
        # Count indicators
        high_count = sum(1 for indicator in high_conf_indicators if indicator in reasoning_lower)
        med_count = sum(1 for indicator in med_conf_indicators if indicator in reasoning_lower)
        low_count = sum(1 for indicator in low_conf_indicators if indicator in reasoning_lower)
        
        # Base confidence by verdict type
        base_confidence = {
            'ENTAILED': 0.7,
            'CONTRADICTED': 0.7, 
            'INSUFFICIENT': 0.3
        }
        
        confidence = base_confidence.get(verdict, 0.5)
        
        # Adjust based on language indicators
        if high_count > 0:
            confidence = min(0.95, confidence + 0.2)
        elif med_count > 0:
            confidence = confidence  # Keep base confidence
        elif low_count > 0:
            confidence = max(0.1, confidence - 0.2)
        
        # Additional adjustments based on verdict-specific language
        if verdict == 'ENTAILED' and any(word in reasoning_lower for word in ['strongly', 'clearly shows', 'evident']):
            confidence = min(0.95, confidence + 0.1)
        elif verdict == 'CONTRADICTED' and any(word in reasoning_lower for word in ['directly opposes', 'contradicts', 'opposite']):
            confidence = min(0.95, confidence + 0.1)
        elif verdict == 'INSUFFICIENT' and any(word in reasoning_lower for word in ['cannot determine', 'not enough', 'lacks']):
            confidence = max(0.1, confidence - 0.1)
        
        return confidence

class VLMVerifier:
    """Main VLM verifier that orchestrates claim generation and verification."""
    
    def __init__(self, vlm: VLMWrapper):
        self.vlm = vlm
        self.claim_generator = MicroClaimGenerator(vlm)
        self.claim_verifier = ClaimVerifier(vlm)
    
    def verify_action_consistency_from_images(self, 
                                     action_description: str,
                                     current_image: str,
                                     previous_image: str,
                                     question: str,
                                     answer_choices: Optional[List[str]] = None,
                                     reason: bool = True,
                                     question_type: Optional[str] = None) -> Dict:
        """
        Verify consistency of an action by comparing two images with confidence-weighted metrics.
        
        Args:
            action_description: Description of the action taken
            current_image: Path to the current image
            previous_image: Path to the previous image
            question: The original spatial reasoning question
            answer_choices: Optional list of answer choices to focus claims on
            reason: Whether to include reasoning in the verification prompt (default: True)
            
        Returns:
            Dictionary containing verification metrics and results
        """
        # Check if both images exist
        if not os.path.exists(current_image) or not os.path.exists(previous_image):
            print(f"One or both images don't exist: {current_image}, {previous_image}")
            return {
            # Original metrics (for backward compatibility)
            'consistency_score': 0.,
            'claim_acceptance_rate': 0.,
            'total_claims': 0.,
            'accepted_claims': 0.,
            'rejected_claims': 0.,
            'uncertain_claims': 0.,
            
            # Enhanced confidence-weighted metrics
            'weighted_accepted': 0.,
            'weighted_rejected': 0.,
            'weighted_insufficient': 0.,
            'average_confidence': 0.,
            'total_confidence': 0.,
            
            # New quality metrics
            'reliability_score': 0.,
            'evidence_quality_score': 0.,
            
            # Raw data
            'claims': None,
            'verification_results': None
        }
        
        # Generate micro-claims based on image comparison
        claims = self.claim_generator._generate_claims_from_images(
            action_description, [previous_image, current_image], question, answer_choices=answer_choices, question_type=question_type
        )
        
        if not claims:
            return {
            # Original metrics (for backward compatibility)
            'confidence_weighted_claim_acceptance_rate': 0.,
            'claim_acceptance_rate': 0.,
            'total_claims': 0.,
            'accepted_claims': 0.,
            'rejected_claims': 0.,
            'uncertain_claims': 0.,
            'average_confidence': 0.,
            'total_confidence': 0.,
            
            # New quality metrics
            'reliability_score': 0.,
            'evidence_quality_score': 0.,
            
            # Raw data
            'claims': None,
            'verification_results': None
        }
        
        # Verify each claim
        verification_results = []
        for claim in claims:
            verification_result = self.claim_verifier.verify_claim(claim, [previous_image, current_image], reason=reason)
            verification_results.append(verification_result)
        
        # Enhanced metrics using confidence scores
        total_claims = len(claims)
        
        # Weighted counts based on confidence
        weighted_accepted = sum(
            result['confidence'] for result in verification_results 
            if result['verdict'] == 'ENTAILED'
        )
        weighted_rejected = sum(
            result['confidence'] for result in verification_results 
            if result['verdict'] == 'CONTRADICTED'
        )
        weighted_insufficient = sum(
            result['confidence'] for result in verification_results 
            if result['verdict'] == 'INSUFFICIENT'
        )
        
        # Binary counts (for backward compatibility)
        accepted_claims = sum(1 for result in verification_results if result['verdict'] == 'ENTAILED')
        rejected_claims = sum(1 for result in verification_results if result['verdict'] == 'CONTRADICTED')
        uncertain_claims = sum(1 for result in verification_results if result['verdict'] == 'INSUFFICIENT')
        
        # Enhanced metrics
        total_confidence = sum(result['confidence'] for result in verification_results)
        average_confidence = total_confidence / total_claims if total_claims > 0 else 0.0
        
        
        # Confidence-weighted claim acceptance rate
        confidence_weighted_claim_acceptance_rate = weighted_accepted / total_claims if total_claims > 0 else 0.0
        claim_acceptance_rate = accepted_claims / total_claims if total_claims > 0 else 0.0

        reliability_score = (weighted_accepted - weighted_rejected) / total_confidence
        
        # Evidence quality score - combines consistency with confidence
        evidence_quality_score = claim_acceptance_rate * average_confidence
        
        return {
            # Original metrics (for backward compatibility)
            'confidence_weighted_claim_acceptance_rate': confidence_weighted_claim_acceptance_rate,
            'claim_acceptance_rate': claim_acceptance_rate,
            'total_claims': total_claims,
            'accepted_claims': accepted_claims,
            'rejected_claims': rejected_claims,
            'uncertain_claims': uncertain_claims,
            
            # Enhanced confidence-weighted metrics
            'average_confidence': average_confidence,
            'total_confidence': total_confidence,
            
            # New quality metrics
            'reliability_score': reliability_score,
            'evidence_quality_score': evidence_quality_score,
            
            # Raw data
            'claims': claims,
            'verification_results': verification_results
        }
    
    
    def _extract_frames_from_video(self, video_path: str, frame_indices: List[int]) -> List[str]:
        """Extract specific frames from video and save as images."""
        if not os.path.exists(video_path):
            print(f"Video path does not exist: {video_path}")
            return []
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return []
        
        # Create output directory for frames
        video_dir = os.path.dirname(video_path)
        frames_dir = os.path.join(video_dir, "verification_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
            else:
                print(f"Could not read frame {frame_idx} from video")
        
        cap.release()
        return frames
    
    def _get_claim_frames(self, claim: Dict, all_frames: List[str], frame_indices: List[int]) -> List[str]:
        """Get frames relevant to a specific claim."""
        # For now, return all frames - could be made more sophisticated
        return all_frames
