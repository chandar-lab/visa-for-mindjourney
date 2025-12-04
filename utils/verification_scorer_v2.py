import os
import json
import re
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class VerificationScorer:
    """Computes helpfulness and exploration scores from verification results."""
    
    def __init__(self, question:str):
        # Initialize neural similarity model
        self.similarity_model = SentenceTransformer('msmarco-distilbert-base-v4')
        if torch.cuda.is_available():
            self.similarity_model = self.similarity_model.to(torch.device('cuda:0'))
        # Cache for embeddings to avoid recomputation
        self._question_embedding = None
        self._claim_embeddings_cache = {}
        self._choice_embeddings_cache = {}
        
        # Question type specific keywords for claim relevance
        self.question_type_keywords = {
            'ego_movement': [
                'camera', 'rotate', 'rotation', 'turn', 'turned', 'left', 'right', 
                'viewpoint', 'perspective', 'angle', 'orientation', 'direction'
            ],
            'obj_movement': [
                'move', 'moved', 'position', 'location', 'object', 'left', 'right', 
                'towards', 'away', 'closer', 'further', 'distance', 'shift'
            ],
            'perspective': [
                'left', 'right', 'relative', 'position', 'side', 'direction', 
                'facing', 'orientation', 'viewpoint', 'perspective'
            ],
            'goal_aim': [
                'direction', 'rotate', 'turn', 'face', 'facing', 'target', 'goal',
                'towards', 'away', 'closer', 'reach', 'aim'
            ],
            'action_conseq': [
                'consequence', 'result', 'outcome', 'effect', 'if', 'then', 'will',
                'closer', 'further', 'facing', 'away', 'towards', 'reach'
            ]
        }
        
        # Spatial relationship keywords for exploration scoring
        self.spatial_keywords = [
            'behind', 'in front', 'left', 'right', 'above', 'below', 'next to',
            'near', 'far', 'between', 'inside', 'outside', 'corner', 'edge',
            'center', 'middle', 'side', 'top', 'bottom', 'visible', 'hidden',
            'occluded', 'clear', 'distance', 'angle', 'depth', 'height'
        ]

        # Novel information indicators
        self.novelty_indicators = [
            'new', 'previously', 'now visible', 'appears', 'disappears',
            'reveals', 'shows', 'hidden', 'occluded', 'clear view',
            'better view', 'different angle', 'closer look'
        ]

        self.question = question["question"]
        self.question_type = question["question_type"]
        
        # Flag to switch between rule-based and neural approaches
        self.use_neural = True  # Set to False to use original rule-based methods
        if self.use_neural:
            print("Using neural mode")
        else:
            print("Using rule-based mode")

    def set_neural_mode(self, use_neural: bool):
        """Switch between neural and rule-based scoring methods."""
        self.use_neural = use_neural

    

    def compute_verification_scores(self, 
                                   verification_results: Dict,
                                   action_description: str) -> Tuple[float, float]:
        """Compute helpfulness and exploration scores from verification results."""
        claims = verification_results.get('claims', [])
        verification_data = verification_results.get('verification_results', [])
        if not claims or not verification_data:
            return 0.0, 0.0
        
        # Extract answer choices from question (simplified)
        answer_choices = self._extract_answer_choices()
        
        # Compute helpfulness score
        helpfulness_score = self._compute_helpfulness_score(
            verification_results, answer_choices
        )
        
        # Compute exploration score
        exploration_score = 0.
        
        return helpfulness_score, exploration_score

    def _extract_answer_choices(self) -> List[str]:
        """Extract answer choices from question text if present."""
        choices = []
        question_text = self.question
        if 'left' in question_text.lower() and 'right' in question_text.lower():
            choices.extend(['left', 'right'])
        if 'yes' in question_text.lower() and 'no' in question_text.lower():
            choices.extend(['yes', 'no'])
        if 'towards' in question_text.lower() and 'away' in question_text.lower():
            choices.extend(['towards', 'away'])
        return choices

    def _compute_helpfulness_score(self, 
                                 verification_results: Dict,
                                 answer_choices: List[str]) -> float:
        """Compute helpfulness score based on question relevance and answer support."""
        verification_data = verification_results.get('verification_results', [])
        # total_score = verification_results["claim_acceptance_rate"] * 10
        # total_score += (verification_results["evidence_quality_score"] * 10)
        # return total_score / 2.0
        # 1. Question Alignment (40% weight)
        # if self.use_neural:
        # question_alignment = self._compute_question_relevance_neural(
        #     verification_results, use_verifier=False
        # )
        # else:
        #     question_alignment = self._compute_question_relevance(
        #         verification_results
        #     )
        
        # # 2. Answer Support (30% weight)
        # if self.use_neural:
        # answer_support = self._compute_answer_support_neural(
        #     verification_results, answer_choices, use_verifier=False
        # )
        # else:
        #     answer_support = self._compute_answer_support(
        #         verification_results, answer_choices
        #     )
        
        evidence_quality = verification_results["evidence_quality_score"]
        # print(f"evidence_quality: {evidence_quality}, question_alignment: {question_alignment}, answer_support: {answer_support}")
        helpfulness_score = evidence_quality * 100 
        # scores = [question_alignment, answer_support]

        # # Weighted combination with normalized scores
        # helpfulness_score = sum(scores) / len(scores) * 10
        
        return helpfulness_score

    def _compute_exploration_score(self,
                                 claims: List[Dict],
                                 verification_results: Dict,
                                 action_description: str) -> float:
        """Compute exploration score based on novel information and coverage."""
        
        verification_data = verification_results.get('verification_results', [])
        # # total_score =  verification_results["average_confidence"] * 10
        # total_score = verification_results["evidence_quality_score"] * 10 
        # return total_score 
        # # 1. Novel Information (35% weight)
        novel_info = self._compute_novel_information(claims, verification_data)
        
        # 2. Coverage Diversity (25% weight)
        coverage_diversity = self._compute_coverage_diversity(claims, verification_data)
        

        # 4. Trajectory Validity (15% weight)
        trajectory_validity = self._compute_trajectory_validity(
            claims, verification_data, action_description
        )
        
        scores = [novel_info, coverage_diversity, trajectory_validity]
        normalized_scores = self._normalize_scores(scores)
        
        # Weighted combination with normalized scores
        exploration_score = (
            0.33 * normalized_scores[0] +  # novel_info
            0.33 * normalized_scores[1] +  # coverage_diversity
            0.33 * normalized_scores[2]    # trajectory_validity
        ) * 10
        
        return min(10.0, max(0.0, exploration_score))


    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to the same effective scale using min-max normalization."""
        if not scores or all(s == 0 for s in scores):
            return scores
        
        # Method 1: Min-Max normalization to [0, 1]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same, return original scores
            return scores
        
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized

    def _compute_answer_support(self, verification_results, answer_choices):
        """How well do the claims support or contradict answer choices?"""
        claims = verification_results.get('claims', [])
        
        support_scores = []
        
        for answer_choice in answer_choices:
            choice_support = 0.0
            choice_confidence = 0.0
            
            for i, claim in enumerate(claims):
                claim_text = claim.get('text', '').lower()
                choice_lower = answer_choice.lower()
                
                # Check if claim supports this choice
                if self._claim_supports_choice(claim_text, choice_lower):
                    confidence = verification_results['verification_results'][i].get('confidence', 0.0)
                    choice_support += confidence
                    choice_confidence += 1.0
            
            if choice_confidence > 0:
                support_scores.append(choice_support / choice_confidence)
                # support_scores.append(choice_confidence)
            else:
                support_scores.append(0.0)
        return max(support_scores) if support_scores else 0.0

    def _compute_answer_support_neural(self, verification_results, answer_choices, use_verifier: bool):
        """How well do the claims support or contradict answer choices using neural similarity?"""
        claims = verification_results.get('claims', [])
        
        support_scores = []
        
        for answer_choice in answer_choices:
            choice_support = 0.0
            choice_confidence = 0.0
            
            for i, claim in enumerate(claims):
                claim_text = claim.get('text', '')
                choice_lower = answer_choice.lower()
                
                # Use neural similarity instead of rule-based matching
                similarity_score = self._claim_supports_choice_neural(claim_text, choice_lower)
                if similarity_score > 0.0:
                    if use_verifier:
                        confidence = verification_results['verification_results'][i].get('confidence', 0.0)
                    else:
                        confidence = 1.0
                    
                    # Weight by similarity score and confidence
                    weighted_support = similarity_score * confidence
                    choice_support += weighted_support
                    choice_confidence += 1.0
            
            if choice_confidence > 0:
                support_scores.append(choice_support / choice_confidence)
            else:
                support_scores.append(0.0)
        
        return max(support_scores) if support_scores else 0.0

    def _claim_supports_choice(self, claim_text: str, choice_text: str) -> bool:
        """Determine if a claim supports a specific answer choice."""
        # Extract key terms from choice
        choice_terms = re.findall(r'\b\w+\b', choice_text.lower())
        choice_terms = [term for term in choice_terms if len(term) > 2]
        
        if not choice_terms:
            return False
        
        # Check for direct term matches
        direct_matches = sum(1 for term in choice_terms if term in claim_text) 
        
        # Check for semantic matches (simplified)
        semantic_matches = 0
        
        # Left/Right semantic matching
        if 'left' in choice_text.lower():
            if any(word in claim_text for word in ['left', 'leftward', 'left side', 'left half']):
                semantic_matches += 1
        elif 'right' in choice_text.lower():
            if any(word in claim_text for word in ['right', 'rightward', 'right side', 'right half']):
                semantic_matches += 1
        
        # Yes/No semantic matching
        if 'yes' in choice_text.lower():
            if any(word in claim_text for word in ['yes', 'true', 'correct', 'affirmative', 'confirmed']):
                semantic_matches += 1
        elif 'no' in choice_text.lower():
            if any(word in claim_text for word in ['no', 'false', 'incorrect', 'negative', 'denied']):
                semantic_matches += 1
        
        # Towards/Away semantic matching
        if 'towards' in choice_text.lower():
            if any(word in claim_text for word in ['towards', 'toward', 'closer', 'approaching', 'nearer']):
                semantic_matches += 1
        elif 'away' in choice_text.lower():
            if any(word in claim_text for word in ['away', 'further', 'receding', 'distant', 'far']):
                semantic_matches += 1
        
        # Consider it supportive if there are either direct matches or semantic matches
        return direct_matches > 0 or semantic_matches > 0


    # Add helper method for choice embeddings:
    def _get_choice_embedding(self, choice_text: str) -> np.ndarray:
        """Get choice embedding, computing only once and caching."""
        if choice_text not in self._choice_embeddings_cache:
            self._choice_embeddings_cache[choice_text] = self.similarity_model.encode([choice_text])
        return self._choice_embeddings_cache[choice_text]

    # Improved _claim_supports_choice_neural method:
    def _claim_supports_choice_neural(self, claim_text: str, choice_text: str) -> float:
        """Determine if a claim supports a specific answer choice using neural similarity."""
        # Get cached choice embedding
        choice_embedding = self._get_choice_embedding(choice_text)
        
        # For claim embedding, we need to check if it's already cached
        # If not, we encode just this single claim
        if claim_text not in self._claim_embeddings_cache:
            self._claim_embeddings_cache[claim_text] = self.similarity_model.encode([claim_text])
        
        claim_embedding = self._claim_embeddings_cache[claim_text]
        
        # Compute similarity
        similarity = cosine_similarity(claim_embedding, choice_embedding)[0][0]
        
        return similarity
        
    def _extract_question_keywords(self, question_text: str) -> List[str]:
        """Extract relevant keywords from the question for matching with claims."""
        question_lower = question_text.lower()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Split into words and filter
        words = re.findall(r'\b\w+\b', question_lower)
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Add question-type specific keywords
        type_keywords = self.question_type_keywords.get(self.question_type, [])
        keywords.extend(type_keywords)
        
        # Add spatial relationship keywords
        keywords.extend(self.spatial_keywords)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords

    def _compute_question_relevance(self, verification_results: Dict) -> float:
        """Compute how relevant the claims are to the specific question."""
        claims = verification_results.get('claims', [])
        verification_data = verification_results.get('verification_results', [])
        
        if not claims or not verification_data:
            return 0.0
        
        # Extract question keywords
        question_keywords = self._extract_question_keywords(self.question)
        
        relevant_claims = 0
        total_confidence = 0.0
        
        for i, claim in enumerate(claims):
            if i >= len(verification_data):
                continue
                
            claim_text = claim.get('text', '').lower()
            verdict = verification_data[i].get('verdict', 'INSUFFICIENT')
            confidence = verification_data[i].get('confidence', 0.0)
            
            if verdict != 'ENTAILED':
                continue
            
            # Check keyword overlap
            keyword_overlap = sum(1 for keyword in question_keywords 
                                if keyword in claim_text)
            
            if keyword_overlap > 0:
                relevant_claims += 1
                # Weight by confidence and keyword overlap ratio
                overlap_ratio = keyword_overlap / len(question_keywords)
                total_confidence += confidence * overlap_ratio
        
        if relevant_claims == 0:
            return 0.0
        
        return total_confidence / relevant_claims

    def _get_question_embedding(self) -> np.ndarray:
        """Get question embedding, computing only once and caching."""
        if self._question_embedding is None:
            self._question_embedding = self.similarity_model.encode([self.question])
        return self._question_embedding

    def _get_claim_embeddings(self, claims: List[Dict]) -> np.ndarray:
        """Get claim embeddings, computing only once and caching."""
        # Create a cache key based on claim texts
        claim_texts = [claim.get('text', '') for claim in claims]
        cache_key = hash(tuple(claim_texts))
        
        if cache_key not in self._claim_embeddings_cache:
            self._claim_embeddings_cache[cache_key] = self.similarity_model.encode(claim_texts)
        
        return self._claim_embeddings_cache[cache_key]

    def _compute_question_relevance_neural(self, verification_results: Dict, use_verifier: bool) -> float:
        """Compute how relevant the claims are to the specific question using neural similarity."""
        claims = verification_results.get('claims', [])
        verification_data = verification_results.get('verification_results', [])
        
        if not claims or not verification_data:
            return 0.0
        
        # Get embeddings efficiently
        question_embedding = self._get_question_embedding()
        claim_embeddings = self._get_claim_embeddings(claims)
        
        # Compute similarities
        similarities = cosine_similarity(question_embedding, claim_embeddings)[0]
        
        relevant_claims = 0
        total_confidence = 0.0
        
        for i, claim in enumerate(claims):
            if i >= len(verification_data):
                continue
            
            confidence = 1.0
            if use_verifier:
                verdict = verification_data[i].get('verdict', 'INSUFFICIENT')
                confidence = verification_data[i].get('confidence', 0.0)
                
                if verdict != 'ENTAILED':
                    continue
            
            # Use neural similarity instead of keyword overlap
            similarity_score = similarities[i]
            
            if similarity_score > 0.0:  # Threshold for relevance
                relevant_claims += 1
                # Weight by confidence and similarity score
                total_confidence += confidence * similarity_score
        
        if relevant_claims == 0:
            return 0.0
        
        return total_confidence / relevant_claims

    def _compute_novel_information(self,
                                claims: List[Dict],
                                verification_data: List[Dict]) -> float:
        """Compute how much novel information the claims provide."""
        if not claims:
            return 0.0
        
        novel_score = 0.0
        valid_claims = 0
        
        for i, claim in enumerate(claims):
            claim_text = claim.get('text', '').lower()
            
            if i >= len(verification_data):
                continue
                
            verdict = verification_data[i].get('verdict', 'INSUFFICIENT')
            confidence = verification_data[i].get('confidence', 0.0)
            
            if verdict != 'ENTAILED':
                continue
            
            # Check for novelty indicators
            novelty_count = sum(1 for indicator in self.novelty_indicators 
                            if indicator in claim_text)
            
            # Additional novelty patterns
            novelty_patterns = [
                r'\b(previously|before)\s+(unseen|hidden|invisible)',
                r'\b(now|currently)\s+(visible|seen|apparent)',
                r'\b(emerges|appears|reveals)\s+(from|out\s+of)',
                r'\b(newly|recently)\s+(discovered|found|identified)',
                r'\b(additional|extra|more)\s+(information|details|clues)',
                r'\b(better|clearer|improved)\s+(view|perspective|angle)',
                r'\b(different|alternative|new)\s+(angle|viewpoint|perspective)',
                r'\b(expanded|broader|wider)\s+(view|field|scope)',
                r'\b(enhanced|improved|better)\s+(visibility|clarity|resolution)',
                r'\b(discovered|found|identified)\s+(new|additional|extra)'
            ]
            
            # Count pattern matches
            pattern_matches = sum(1 for pattern in novelty_patterns 
                                if re.search(pattern, claim_text))
            
            # Combine keyword and pattern novelty
            total_novelty = novelty_count + pattern_matches
            
            if total_novelty > 0:
                # Weight by confidence and normalize by maximum possible novelty
                max_possible_novelty = len(self.novelty_indicators) + len(novelty_patterns)
                novelty_ratio = min(total_novelty / max_possible_novelty, 1.0)
                novel_score += confidence * novelty_ratio
                valid_claims += 1
        
        return novel_score / max(valid_claims, 1)


    def _compute_coverage_diversity(self,
                                claims: List[Dict],
                                verification_data: List[Dict]) -> float:
        """Compute diversity of spatial aspects covered by claims."""
        if not claims:
            return 0.0
        
        covered_aspects = set()
        total_confidence = 0.0
        valid_claims = 0
        
        for i, claim in enumerate(claims):
            claim_text = claim.get('text', '').lower()
            
            if i >= len(verification_data):
                continue
                
            verdict = verification_data[i].get('verdict', 'INSUFFICIENT')
            confidence = verification_data[i].get('confidence', 0.0)
            
            if verdict != 'ENTAILED':
                continue
            
            # Categorize spatial aspects based on claim content
            aspect_confidence = 0.0
            
            # 1. Depth/Distance aspects
            if any(word in claim_text for word in ['depth', 'distance', 'far', 'close', 'near', 'away', 'closer', 'further', 'proximity']):
                covered_aspects.add('depth')
                aspect_confidence = max(aspect_confidence, confidence)
            
            # 2. Angular/Rotational aspects
            if any(word in claim_text for word in ['angle', 'rotation', 'turn', 'left', 'right', 'clockwise', 'counterclockwise', 'degree', 'orientation']):
                covered_aspects.add('angle')
                aspect_confidence = max(aspect_confidence, confidence)
            
            # 3. Occlusion/Visibility aspects
            if any(word in claim_text for word in ['occlusion', 'hidden', 'visible', 'blocked', 'obstructed', 'covered', 'uncovered', 'revealed', 'concealed']):
                covered_aspects.add('occlusion')
                aspect_confidence = max(aspect_confidence, confidence)
            
            # 4. Height/Vertical aspects
            if any(word in claim_text for word in ['height', 'above', 'below', 'top', 'bottom', 'vertical', 'elevated', 'lowered', 'up', 'down']):
                covered_aspects.add('height')
                aspect_confidence = max(aspect_confidence, confidence)
            
            # 5. Horizontal positioning aspects
            if any(word in claim_text for word in ['side', 'beside', 'next to', 'adjacent', 'lateral', 'horizontal', 'parallel', 'perpendicular']):
                covered_aspects.add('horizontal')
                aspect_confidence = max(aspect_confidence, confidence)
            
            # 6. Relative positioning aspects
            if any(word in claim_text for word in ['behind', 'in front', 'between', 'inside', 'outside', 'center', 'middle', 'edge', 'corner']):
                covered_aspects.add('relative_position')
                aspect_confidence = max(aspect_confidence, confidence)
            
            # 7. Temporal/Change aspects (for dynamic scenes)
            if any(word in claim_text for word in ['change', 'moved', 'shifted', 'appeared', 'disappeared', 'before', 'after', 'now', 'previously']):
                covered_aspects.add('temporal_change')
                aspect_confidence = max(aspect_confidence, confidence)
            
            # 8. Object properties aspects
            if any(word in claim_text for word in ['size', 'shape', 'color', 'texture', 'material', 'dimension', 'scale', 'proportion']):
                covered_aspects.add('object_properties')
                aspect_confidence = max(aspect_confidence, confidence)
            
            total_confidence += aspect_confidence
            valid_claims += 1
        
        if valid_claims == 0:
            return 0.0
        
        # Calculate diversity metrics
        max_aspects = 8  # Total number of aspect categories
        diversity_ratio = len(covered_aspects) / max_aspects
        
        # Average confidence weighted by aspect coverage
        avg_confidence = total_confidence / valid_claims
        
        # Bonus for covering multiple aspects in a single claim
        multi_aspect_bonus = 0.0
        for i, claim in enumerate(claims):
            if i >= len(verification_data):
                continue
            verdict = verification_data[i].get('verdict', 'INSUFFICIENT')
            if verdict != 'ENTAILED':
                continue
                
            claim_text = claim.get('text', '').lower()
            aspects_in_claim = 0
            
            # Count how many aspects this single claim covers
            aspect_keywords = {
                'depth': ['depth', 'distance', 'far', 'close', 'near', 'away'],
                'angle': ['angle', 'rotation', 'turn', 'left', 'right'],
                'occlusion': ['occlusion', 'hidden', 'visible', 'blocked'],
                'height': ['height', 'above', 'below', 'top', 'bottom'],
                'horizontal': ['side', 'beside', 'next to', 'adjacent'],
                'relative_position': ['behind', 'in front', 'between'],
                'temporal_change': ['change', 'moved', 'shifted', 'appeared'],
                'object_properties': ['size', 'shape', 'color', 'texture']
            }
            
            for aspect, keywords in aspect_keywords.items():
                if any(keyword in claim_text for keyword in keywords):
                    aspects_in_claim += 1
            
            # Reward claims that cover multiple aspects
            if aspects_in_claim > 1:
                multi_aspect_bonus += 0.1 * (aspects_in_claim - 1)
        
        # Normalize multi-aspect bonus
        multi_aspect_bonus = min(multi_aspect_bonus, 0.3)  # Cap at 30% bonus
        
        # Final diversity score
        diversity_score = diversity_ratio * avg_confidence + multi_aspect_bonus
        
        return min(1.0, diversity_score)

    def _compute_trajectory_validity(self,
                                claims: List[Dict],
                                verification_data: List[Dict],
                                action_description: str) -> float:
        """Compute validity of the camera trajectory based on claims."""
        if not claims:
            return 0.0
        
        validity_score = 0.0
        total_claims = 0
        
        action_lower = action_description.lower()
        
        # Define expected outcomes for different actions
        action_outcomes = {
            'move forward': {
                'positive_indicators': [
                    'closer', 'nearer', 'approaching', 'advancing', 'forward',
                    'distance reduced', 'getting closer', 'coming nearer'
                ],
                'negative_indicators': [
                    'further', 'away', 'receding', 'distance increased', 'backing away'
                ]
            },
            'turn left': {
                'positive_indicators': [
                    'left', 'rotated left', 'turned left', 'clockwise', 'counter-clockwise',
                    'leftward', 'to the left', 'left side', 'leftward rotation'
                ],
                'negative_indicators': [
                    'right', 'rotated right', 'turned right', 'rightward', 'to the right'
                ]
            },
            'turn right': {
                'positive_indicators': [
                    'right', 'rotated right', 'turned right', 'clockwise', 'rightward',
                    'to the right', 'right side', 'rightward rotation'
                ],
                'negative_indicators': [
                    'left', 'rotated left', 'turned left', 'leftward', 'to the left'
                ]
            }
        }
        
        # Find matching action type
        matching_action = None
        for action_type, outcomes in action_outcomes.items():
            if action_type in action_lower:
                matching_action = outcomes
                break
        
        if not matching_action:
            # Unknown action type - return neutral score
            return 0.5
        
        positive_indicators = matching_action['positive_indicators']
        negative_indicators = matching_action['negative_indicators']
        
        for i, claim in enumerate(claims):
            claim_text = claim.get('text', '').lower()
            
            if i >= len(verification_data):
                continue
                
            verdict = verification_data[i].get('verdict', 'INSUFFICIENT')
            confidence = verification_data[i].get('confidence', 0.0)
            
            if verdict != 'ENTAILED':
                continue
            
            # Check if claim validates the expected action outcome
            claim_validity = 0.0
            
            # Count positive indicator matches
            positive_matches = sum(1 for indicator in positive_indicators 
                                if indicator in claim_text)
            
            # Count negative indicator matches
            negative_matches = sum(1 for indicator in negative_indicators 
                                if indicator in claim_text)
            
            # Calculate validity score for this claim
            if positive_matches > 0 and negative_matches == 0:
                # Strong positive validation
                claim_validity = confidence * 1.0
            elif positive_matches > negative_matches:
                # Weak positive validation
                claim_validity = confidence * 0.7
            elif positive_matches == negative_matches and positive_matches > 0:
                # Neutral (contradictory indicators)
                claim_validity = confidence * 0.3
            elif negative_matches > positive_matches:
                # Negative validation (action didn't work as expected)
                claim_validity = confidence * 0.1
            else:
                # No clear indicators
                claim_validity = confidence * 0.2
            
            # Additional bonus for specific spatial validation
            spatial_validation_bonus = self._compute_spatial_validation_bonus(
                claim_text, action_description, matching_action
            )
            claim_validity += spatial_validation_bonus * confidence
            
            validity_score += claim_validity
            total_claims += 1
        
        return validity_score / max(total_claims, 1)

    def _compute_spatial_validation_bonus(self, 
                                        claim_text: str, 
                                        action_description: str,
                                        expected_outcomes: Dict) -> float:
        """Compute bonus for specific spatial validation patterns."""
        bonus = 0.0
        
        # Bonus for magnitude-specific validation
        if 'degree' in action_description or 'angle' in action_description:
            if any(word in claim_text for word in ['angle', 'rotation', 'degree', 'turn']):
                bonus += 0.2
        
        if 'meter' in action_description or 'distance' in action_description:
            if any(word in claim_text for word in ['distance', 'closer', 'further', 'meter']):
                bonus += 0.2
        
        # Bonus for temporal validation (before/after comparisons)
        if any(word in claim_text for word in ['now', 'currently', 'after', 'previously', 'before']):
            bonus += 0.1
        
        # Bonus for object-specific validation
        if any(word in claim_text for word in ['object', 'item', 'thing', 'furniture', 'wall']):
            bonus += 0.1
        
        # Bonus for directional precision
        if any(word in claim_text for word in ['precisely', 'exactly', 'directly', 'straight']):
            bonus += 0.15
        
        return min(bonus, 0.5)  # Cap bonus at 0.5

    def _validate_action_physics(self, 
                            claims: List[Dict],
                            verification_data: List[Dict],
                            action_description: str) -> float:
        """Validate that the action follows physical constraints."""
        physics_score = 0.0
        valid_claims = 0
        
        for i, claim in enumerate(claims):
            claim_text = claim.get('text', '').lower()
            
            if i >= len(verification_data):
                continue
                
            verdict = verification_data[i].get('verdict', 'INSUFFICIENT')
            confidence = verification_data[i].get('confidence', 0.0)
            
            if verdict != 'ENTAILED':
                continue
            
            # Check for physically plausible outcomes
            physics_indicators = [
                'natural', 'realistic', 'consistent', 'logical', 'reasonable',
                'expected', 'normal', 'typical', 'standard', 'conventional'
            ]
            
            physics_violations = [
                'impossible', 'unrealistic', 'inconsistent', 'illogical', 'unreasonable',
                'unexpected', 'abnormal', 'atypical', 'impossible', 'unconventional'
            ]
            
            has_physics_indicators = any(indicator in claim_text for indicator in physics_indicators)
            has_physics_violations = any(violation in claim_text for violation in physics_violations)
            
            if has_physics_indicators and not has_physics_violations:
                physics_score += confidence * 1.0
            elif not has_physics_violations:
                physics_score += confidence * 0.5
            else:
                physics_score += confidence * 0.1
            
            valid_claims += 1
        
        return physics_score / max(valid_claims, 1)

if __name__ == "__main__":
    scorer = VerificationScorer({"question": "What is the color of the mug?",
                                "question_type": "color"})
    print(scorer.compute_verification_scores({"claims": [], "verification_results": []}, "move forward"))