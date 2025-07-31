"""
BLEND Proof-of-Forecast Consensus Protocol Implementation
Novel consensus mechanism based on predictive accuracy

Author: Raed Abdel-Sater
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import time
import json
import logging
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from ..utils import MetricsCalculator


class ConsensusState(Enum):
    """Consensus protocol states"""
    INITIALIZING = "initializing"
    PROPOSAL_COLLECTION = "proposal_collection"
    EVALUATION_PHASE = "evaluation_phase"
    VOTING_PHASE = "voting_phase"
    CONSENSUS_REACHED = "consensus_reached"
    CONSENSUS_FAILED = "consensus_failed"


@dataclass
class ProofOfForecastResult:
    """Result of Proof-of-Forecast consensus"""
    consensus_reached: bool
    selected_proposal_id: Optional[str]
    selected_proposal: Optional[Dict]
    winning_score: float
    participating_miners: List[str]
    participating_evaluators: List[str]
    consensus_round: int
    timestamp: float
    proof_hash: str
    byzantine_detected: List[str]


class ProofOfForecastConsensus:
    """
    Proof-of-Forecast consensus protocol implementation
    
    This consensus mechanism treats predictive accuracy as the scarce resource
    governing block production, ensuring that only high-quality model updates
    are accepted into the global model.
    """
    
    def __init__(
        self,
        threshold: float = 0.67,
        max_debate_rounds: int = 10,
        blockchain_client = None,
        config: Optional[Dict] = None
    ):
        self.threshold = threshold
        self.max_debate_rounds = max_debate_rounds
        self.blockchain_client = blockchain_client
        self.config = config or {}
        
        # Consensus state
        self.current_state = ConsensusState.INITIALIZING
        self.current_round = 0
        self.consensus_history = []
        
        # Metrics and validation
        self.metrics_calculator = MetricsCalculator()
        self.byzantine_detector = ByzantineDetector(threshold=2.0)  # 2 std dev threshold
        
        # Logger
        self.logger = logging.getLogger('BLEND.ProofOfForecast')
        
    def reach_consensus(
        self,
        proposals: Dict[str, Dict],
        evaluation_scores: Dict[str, Dict],
        current_round: int,
        validation_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute Proof-of-Forecast consensus protocol
        
        Args:
            proposals: Dictionary of miner proposals
            evaluation_scores: Evaluation scores from evaluators
            current_round: Current training round
            validation_data: Optional validation data for additional verification
            
        Returns:
            Consensus result dictionary
        """
        self.current_round = current_round
        self.logger.info(f"Starting PoF consensus for round {current_round}")
        
        # Initialize consensus round
        consensus_start_time = time.time()
        
        try:
            # Phase 1: Validate proposals and evaluations
            validated_proposals, validated_evaluations = self._validate_inputs(
                proposals, evaluation_scores
            )
            
            if not validated_proposals:
                return self._create_failure_result("No valid proposals received")
            
            # Phase 2: Detect Byzantine behavior
            byzantine_miners, byzantine_evaluators = self._detect_byzantine_behavior(
                validated_proposals, validated_evaluations
            )
            
            # Phase 3: Calculate forecast proofs
            forecast_proofs = self._calculate_forecast_proofs(
                validated_proposals, validated_evaluations, validation_data
            )
            
            # Phase 4: Conduct weighted voting
            voting_result = self._conduct_weighted_voting(
                forecast_proofs, byzantine_miners, byzantine_evaluators
            )
            
            # Phase 5: Verify consensus threshold
            if voting_result['consensus_achieved']:
                # Phase 6: Create and validate proof
                proof_hash = self._create_consensus_proof(
                    voting_result, validated_proposals, validated_evaluations
                )
                
                # Create success result
                result = ProofOfForecastResult(
                    consensus_reached=True,
                    selected_proposal_id=voting_result['winner_id'],
                    selected_proposal=validated_proposals[voting_result['winner_id']],
                    winning_score=voting_result['winning_score'],
                    participating_miners=list(validated_proposals.keys()),
                    participating_evaluators=list(validated_evaluations.keys()),
                    consensus_round=current_round,
                    timestamp=time.time(),
                    proof_hash=proof_hash,
                    byzantine_detected=byzantine_miners + byzantine_evaluators
                )
                
                self.logger.info(f"PoF consensus reached in {time.time() - consensus_start_time:.2f}s")
                
            else:
                result = self._create_failure_result(
                    "Consensus threshold not met",
                    byzantine_detected=byzantine_miners + byzantine_evaluators
                )
            
            # Record consensus attempt
            self.consensus_history.append(result)
            
            return self._result_to_dict(result)
            
        except Exception as e:
            self.logger.error(f"Consensus failed with error: {e}")
            return self._create_failure_result(f"Consensus error: {str(e)}")
    
    def _validate_inputs(
        self,
        proposals: Dict[str, Dict],
        evaluation_scores: Dict[str, Dict]
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Validate proposals and evaluation scores"""
        
        validated_proposals = {}
        validated_evaluations = {}
        
        # Validate proposals
        for miner_id, proposal in proposals.items():
            if self._is_valid_proposal(proposal):
                validated_proposals[miner_id] = proposal
            else:
                self.logger.warning(f"Invalid proposal from miner {miner_id}")
        
        # Validate evaluations
        for evaluator_id, scores in evaluation_scores.items():
            if self._is_valid_evaluation(scores):
                validated_evaluations[evaluator_id] = scores
            else:
                self.logger.warning(f"Invalid evaluation from evaluator {evaluator_id}")
        
        self.logger.info(f"Validated {len(validated_proposals)} proposals and {len(validated_evaluations)} evaluations")
        
        return validated_proposals, validated_evaluations
    
    def _is_valid_proposal(self, proposal: Dict) -> bool:
        """Check if proposal is valid"""
        required_fields = ['miner_id', 'model_update', 'training_loss', 'timestamp', 'signature']
        
        # Check required fields
        for field in required_fields:
            if field not in proposal:
                return False
        
        # Check data types and ranges
        if not isinstance(proposal['training_loss'], (int, float)) or proposal['training_loss'] < 0:
            return False
        
        if not isinstance(proposal['timestamp'], (int, float)):
            return False
        
        # Verify signature (simplified)
        return self._verify_proposal_signature(proposal)
    
    def _is_valid_evaluation(self, evaluation: Dict) -> bool:
        """Check if evaluation is valid"""
        if not isinstance(evaluation, dict):
            return False
        
        # Check that all scores are numeric and in valid range
        for miner_id, score_data in evaluation.items():
            if isinstance(score_data, dict) and 'combined_score' in score_data:
                score = score_data['combined_score']
                if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                    return False
        
        return True
    
    def _verify_proposal_signature(self, proposal: Dict) -> bool:
        """Verify proposal signature (simplified implementation)"""
        if 'signature' not in proposal:
            return False
        
        # Create proposal copy without signature for verification
        proposal_copy = proposal.copy()
        signature = proposal_copy.pop('signature')
        
        # Generate expected signature
        proposal_str = json.dumps(proposal_copy, sort_keys=True, default=str)
        expected_signature = hashlib.sha256(
            f"{proposal_str}_{proposal.get('miner_id', '')}".encode()
        ).hexdigest()
        
        return signature == expected_signature
    
    def _detect_byzantine_behavior(
        self,
        proposals: Dict[str, Dict],
        evaluations: Dict[str, Dict]
    ) -> Tuple[List[str], List[str]]:
        """Detect Byzantine (malicious) miners and evaluators"""
        
        byzantine_miners = []
        byzantine_evaluators = []
        
        # Detect Byzantine miners based on proposal anomalies
        training_losses = [p['training_loss'] for p in proposals.values()]
        if len(training_losses) > 2:
            outlier_miners = self.byzantine_detector.detect_outliers(
                data=training_losses,
                identifiers=list(proposals.keys())
            )
            byzantine_miners.extend(outlier_miners)
        
        # Detect Byzantine evaluators based on scoring patterns
        if len(evaluations) > 1:
            # Analyze evaluator agreement
            evaluator_scores = self._extract_evaluator_score_matrix(evaluations)
            if evaluator_scores:
                outlier_evaluators = self._detect_evaluation_outliers(evaluator_scores)
                byzantine_evaluators.extend(outlier_evaluators)
        
        if byzantine_miners:
            self.logger.warning(f"Detected Byzantine miners: {byzantine_miners}")
        if byzantine_evaluators:
            self.logger.warning(f"Detected Byzantine evaluators: {byzantine_evaluators}")
        
        return byzantine_miners, byzantine_evaluators
    
    def _extract_evaluator_score_matrix(self, evaluations: Dict[str, Dict]) -> Dict[str, List[float]]:
        """Extract score matrix for evaluator analysis"""
        score_matrix = {}
        
        # Get all unique miner IDs
        all_miners = set()
        for evaluation in evaluations.values():
            all_miners.update(evaluation.keys())
        
        # Build score matrix
        for evaluator_id, evaluation in evaluations.items():
            scores = []
            for miner_id in sorted(all_miners):
                if miner_id in evaluation and isinstance(evaluation[miner_id], dict):
                    score = evaluation[miner_id].get('combined_score', 0.0)
                    scores.append(score)
                else:
                    scores.append(0.0)  # Default score for missing evaluations
            
            score_matrix[evaluator_id] = scores
        
        return score_matrix
    
    def _detect_evaluation_outliers(self, score_matrix: Dict[str, List[float]]) -> List[str]:
        """Detect outlier evaluators based on scoring patterns"""
        outliers = []
        
        if len(score_matrix) < 3:
            return outliers  # Need at least 3 evaluators for meaningful outlier detection
        
        evaluator_ids = list(score_matrix.keys())
        score_arrays = np.array(list(score_matrix.values()))
        
        # Calculate pairwise correlations between evaluators
        correlations = np.corrcoef(score_arrays)
        
        # Find evaluators with consistently low correlation with others
        for i, evaluator_id in enumerate(evaluator_ids):
            other_correlations = np.concatenate([correlations[i, :i], correlations[i, i+1:]])
            avg_correlation = np.mean(other_correlations)
            
            # If average correlation is very low, mark as potential Byzantine
            if avg_correlation < 0.3:  # Threshold for Byzantine detection
                outliers.append(evaluator_id)
        
        return outliers
    
    def _calculate_forecast_proofs(
        self,
        proposals: Dict[str, Dict],
        evaluations: Dict[str, Dict],
        validation_data: Optional[Dict] = None
    ) -> Dict[str, Dict]:
        """Calculate forecast proofs for each proposal"""
        
        forecast_proofs = {}
        
        for miner_id, proposal in proposals.items():
            # Base proof from training loss
            training_loss = proposal['training_loss']
            base_proof = self._loss_to_proof_score(training_loss)
            
            # Aggregate evaluation scores
            evaluation_proof = self._aggregate_evaluation_scores(miner_id, evaluations)
            
            # Optional validation proof
            validation_proof = 0.0
            if validation_data:
                validation_proof = self._calculate_validation_proof(proposal, validation_data)
            
            # Stake and reputation factors
            stake_factor = self._calculate_stake_factor(proposal.get('stake', 0))
            reputation_factor = self._calculate_reputation_factor(proposal.get('reputation', 1.0))
            
            # Combined proof score
            combined_proof = (
                0.4 * base_proof +
                0.4 * evaluation_proof +
                0.1 * validation_proof +
                0.05 * stake_factor +
                0.05 * reputation_factor
            )
            
            forecast_proofs[miner_id] = {
                'base_proof': base_proof,
                'evaluation_proof': evaluation_proof,
                'validation_proof': validation_proof,
                'stake_factor': stake_factor,
                'reputation_factor': reputation_factor,
                'combined_proof': combined_proof,
                'timestamp': time.time()
            }
        
        return forecast_proofs
    
    def _loss_to_proof_score(self, training_loss: float) -> float:
        """Convert training loss to proof score (0-1 range)"""
        # Use exponential decay: score = exp(-loss)
        return np.exp(-max(0, training_loss))
    
    def _aggregate_evaluation_scores(self, miner_id: str, evaluations: Dict[str, Dict]) -> float:
        """Aggregate evaluation scores for a specific miner"""
        scores = []
        weights = []
        
        for evaluator_id, evaluation in evaluations.items():
            if miner_id in evaluation and isinstance(evaluation[miner_id], dict):
                score = evaluation[miner_id].get('combined_score', 0.0)
                weight = 1.0  # Could weight by evaluator reputation
                
                scores.append(score)
                weights.append(weight)
        
        if not scores:
            return 0.0
        
        # Weighted average
        return np.average(scores, weights=weights)
    
    def _calculate_validation_proof(self, proposal: Dict, validation_data: Dict) -> float:
        """Calculate validation proof (simplified implementation)"""
        # In practice, would load the model and evaluate on validation data
        # For now, return a score based on proposal quality indicators
        
        training_loss = proposal.get('training_loss', float('inf'))
        if training_loss == float('inf'):
            return 0.0
        
        # Simple heuristic: better training loss suggests better validation performance
        return min(1.0, np.exp(-training_loss * 0.5))
    
    def _calculate_stake_factor(self, stake: float) -> float:
        """Calculate stake factor for proof score"""
        if stake <= 0:
            return 0.0
        
        # Logarithmic scaling to prevent stake dominance
        return min(1.0, np.log(1 + stake) / np.log(1 + 1000))
    
    def _calculate_reputation_factor(self, reputation: float) -> float:
        """Calculate reputation factor for proof score"""
        return min(1.0, max(0.0, reputation))
    
    def _conduct_weighted_voting(
        self,
        forecast_proofs: Dict[str, Dict],
        byzantine_miners: List[str],
        byzantine_evaluators: List[str]
    ) -> Dict[str, Any]:
        """Conduct weighted voting based on forecast proofs"""
        
        # Filter out Byzantine actors
        valid_proofs = {
            miner_id: proof for miner_id, proof in forecast_proofs.items()
            if miner_id not in byzantine_miners
        }
        
        if not valid_proofs:
            return {
                'consensus_achieved': False,
                'reason': 'No valid proofs after Byzantine filtering'
            }
        
        # Find highest proof score
        winner_id = max(valid_proofs.keys(), key=lambda x: valid_proofs[x]['combined_proof'])
        winning_score = valid_proofs[winner_id]['combined_proof']
        
        # Check if winner meets minimum threshold
        min_proof_threshold = self.config.get('min_proof_threshold', 0.1)
        if winning_score < min_proof_threshold:
            return {
                'consensus_achieved': False,
                'reason': f'Winning score {winning_score:.4f} below threshold {min_proof_threshold}'
            }
        
        # Calculate voting weight (based on stake and reputation)
        total_voting_weight = sum(
            forecast_proofs[miner_id].get('stake_factor', 0) +
            forecast_proofs[miner_id].get('reputation_factor', 0)
            for miner_id in valid_proofs.keys()
        )
        
        winner_weight = (
            valid_proofs[winner_id].get('stake_factor', 0) +
            valid_proofs[winner_id].get('reputation_factor', 0)
        )
        
        # Check consensus threshold
        if total_voting_weight > 0:
            vote_percentage = winner_weight / total_voting_weight
        else:
            vote_percentage = 1.0 / len(valid_proofs)  # Equal weight fallback
        
        consensus_achieved = vote_percentage >= self.threshold
        
        return {
            'consensus_achieved': consensus_achieved,
            'winner_id': winner_id,
            'winning_score': winning_score,
            'vote_percentage': vote_percentage,
            'total_participants': len(valid_proofs),
            'byzantine_filtered': len(byzantine_miners)
        }
    
    def _create_consensus_proof(
        self,
        voting_result: Dict,
        proposals: Dict[str, Dict],
        evaluations: Dict[str, Dict]
    ) -> str:
        """Create cryptographic proof of consensus"""
        
        proof_data = {
            'voting_result': voting_result,
            'winner_proposal_hash': self._hash_proposal(proposals[voting_result['winner_id']]),
            'evaluation_hashes': {
                evaluator_id: self._hash_evaluation(evaluation)
                for evaluator_id, evaluation in evaluations.items()
            },
            'consensus_round': self.current_round,
            'timestamp': time.time(),
            'protocol_version': '1.0'
        }
        
        # Generate proof hash
        proof_str = json.dumps(proof_data, sort_keys=True, default=str)
        proof_hash = hashlib.sha256(proof_str.encode()).hexdigest()
        
        # Store proof on blockchain if available
        if self.blockchain_client:
            self.blockchain_client.submit_transaction({
                'type': 'consensus_proof',
                'proof_hash': proof_hash,
                'proof_data': proof_data
            })
        
        return proof_hash
    
    def _hash_proposal(self, proposal: Dict) -> str:
        """Generate hash of proposal"""
        proposal_str = json.dumps(proposal, sort_keys=True, default=str)
        return hashlib.sha256(proposal_str.encode()).hexdigest()
    
    def _hash_evaluation(self, evaluation: Dict) -> str:
        """Generate hash of evaluation"""
        evaluation_str = json.dumps(evaluation, sort_keys=True, default=str)
        return hashlib.sha256(evaluation_str.encode()).hexdigest()
    
    def _create_failure_result(
        self,
        reason: str,
        byzantine_detected: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create consensus failure result"""
        
        return {
            'consensus_reached': False,
            'reason': reason,
            'byzantine_detected': byzantine_detected or [],
            'consensus_round': self.current_round,
            'timestamp': time.time()
        }
    
    def _result_to_dict(self, result: ProofOfForecastResult) -> Dict[str, Any]:
        """Convert result dataclass to dictionary"""
        return {
            'consensus_reached': result.consensus_reached,
            'selected_proposal_id': result.selected_proposal_id,
            'selected_proposal': result.selected_proposal,
            'winning_score': result.winning_score,
            'participating_miners': result.participating_miners,
            'participating_evaluators': result.participating_evaluators,
            'consensus_round': result.consensus_round,
            'timestamp': result.timestamp,
            'proof_hash': result.proof_hash,
            'byzantine_detected': result.byzantine_detected
        }
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Get consensus protocol statistics"""
        if not self.consensus_history:
            return {'message': 'No consensus history available'}
        
        successful_rounds = [r for r in self.consensus_history if r.consensus_reached]
        
        return {
            'total_rounds': len(self.consensus_history),
            'successful_rounds': len(successful_rounds),
            'success_rate': len(successful_rounds) / len(self.consensus_history),
            'average_winning_score': np.mean([r.winning_score for r in successful_rounds]) if successful_rounds else 0,
            'total_byzantine_detected': sum(len(r.byzantine_detected) for r in self.consensus_history),
            'average_participants': np.mean([len(r.participating_miners) for r in self.consensus_history])
        }


class ByzantineDetector:
    """Byzantine behavior detection utility"""
    
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold  # Standard deviations for outlier detection
        
    def detect_outliers(self, data: List[float], identifiers: List[str]) -> List[str]:
        """Detect outliers using statistical methods"""
        if len(data) < 3:
            return []  # Need at least 3 data points
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        if std == 0:
            return []  # No variance, no outliers
        
        # Calculate z-scores
        z_scores = np.abs((data_array - mean) / std)
        
        # Identify outliers
        outlier_indices = np.where(z_scores > self.threshold)[0]
        
        return [identifiers[i] for i in outlier_indices]
    
    def detect_collusion(self, score_matrix: Dict[str, List[float]]) -> List[str]:
        """Detect collusive behavior among evaluators"""
        # Simplified collusion detection
        # In practice, would use more sophisticated methods
        
        if len(score_matrix) < 3:
            return []
        
        evaluator_ids = list(score_matrix.keys())
        correlations = []
        
        # Calculate all pairwise correlations
        for i in range(len(evaluator_ids)):
            for j in range(i + 1, len(evaluator_ids)):
                scores_i = score_matrix[evaluator_ids[i]]
                scores_j = score_matrix[evaluator_ids[j]]
                
                correlation = np.corrcoef(scores_i, scores_j)[0, 1]
                if not np.isnan(correlation):
                    correlations.append((evaluator_ids[i], evaluator_ids[j], correlation))
        
        # Identify suspiciously high correlations (potential collusion)
        collusive_pairs = [
            (eval_i, eval_j) for eval_i, eval_j, corr in correlations 
            if corr > 0.95  # Very high correlation threshold
        ]
        
        # Extract unique evaluators involved in collusion
        collusive_evaluators = set()
        for eval_i, eval_j in collusive_pairs:
            collusive_evaluators.add(eval_i)
            collusive_evaluators.add(eval_j)
        
        return list(collusive_evaluators)


# Utility functions for consensus protocol
def create_proof_of_forecast_consensus(
    threshold: float = 0.67,
    max_debate_rounds: int = 10,
    blockchain_client = None
) -> ProofOfForecastConsensus:
    """Factory function to create PoF consensus protocol"""
    return ProofOfForecastConsensus(
        threshold=threshold,
        max_debate_rounds=max_debate_rounds,
        blockchain_client=blockchain_client
    )


def validate_consensus_proof(proof_hash: str, proof_data: Dict) -> bool:
    """Validate a consensus proof"""
    # Regenerate proof hash and compare
    proof_str = json.dumps(proof_data, sort_keys=True, default=str)
    expected_hash = hashlib.sha256(proof_str.encode()).hexdigest()
    
    return proof_hash == expected_hash