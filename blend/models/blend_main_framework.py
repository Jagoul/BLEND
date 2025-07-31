"""
BLEND: Blockchain-Enhanced Network Decentralisation with LLMs
Main Framework Implementation

Author: Raed Abdel-Sater
Institution: Concordia Institute for Information Systems Engineering
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import yaml
import logging
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .agents import Publisher, Oracle, Miner, Evaluator, Leader
from .blockchain import ProofOfForecastConsensus, BlockchainClient
from .models import TaskAlignedLLM, QLoRAAdapter
from .federated import FedADAMOptimizer, ModelAggregation
from .incentives import RewardSystem, StakeDynamics
from .utils import DataLoader, MetricsCalculator, ConfigManager


@dataclass
class BLENDConfig:
    """Configuration class for BLEND framework"""
    # Model parameters
    model_name: str = "qwen-3-8b"
    model_path: Optional[str] = None
    qlora_rank: int = 16
    qlora_alpha: int = 32
    quantization_bits: int = 4
    
    # Training parameters
    local_epochs: int = 3
    global_rounds: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    fedadam_beta1: float = 0.9
    fedadam_beta2: float = 0.999
    
    # Blockchain parameters
    consensus_threshold: float = 0.67
    max_debate_rounds: int = 10
    block_generation_rate: float = 1.3  # seconds
    
    # Agent parameters
    num_miners: int = 10
    num_evaluators: int = 3
    oracle_update_frequency: int = 5
    
    # Incentive parameters
    performance_reward_scale: float = 1.0
    evaluation_reward_scale: float = 0.5
    oracle_reward_scale: float = 0.3
    leader_reward_scale: float = 0.2
    stake_decay_rate: float = 0.01
    
    # Forecasting parameters
    lookback_window: int = 720
    prediction_horizons: List[int] = None
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [96, 192, 336, 720]


class BLENDFramework:
    """
    Main BLEND Framework class implementing blockchain-enhanced 
    federated learning for time-series forecasting
    """
    
    def __init__(
        self,
        config: Union[BLENDConfig, str, Dict],
        device: Optional[torch.device] = None
    ):
        """
        Initialize BLEND framework
        
        Args:
            config: Configuration object, path to config file, or config dict
            device: PyTorch device for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config(config)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.blockchain_client = None
        self.consensus_protocol = None
        self.agents = {}
        self.global_model = None
        self.reward_system = None
        self.stake_dynamics = None
        
        # Training state
        self.current_round = 0
        self.training_history = []
        self.performance_metrics = {}
        
        self.logger.info("BLEND Framework initialized successfully")
    
    def _load_config(self, config: Union[BLENDConfig, str, Dict]) -> BLENDConfig:
        """Load configuration from various sources"""
        if isinstance(config, BLENDConfig):
            return config
        elif isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
            return BLENDConfig(**config_dict)
        elif isinstance(config, dict):
            return BLENDConfig(**config)
        else:
            raise ValueError("Invalid config type")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('BLEND')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def setup_blockchain(self, fabric_config_path: Optional[str] = None) -> None:
        """Initialize blockchain infrastructure"""
        self.logger.info("Setting up blockchain infrastructure...")
        
        # Initialize Hyperledger Fabric client
        self.blockchain_client = BlockchainClient(
            config_path=fabric_config_path,
            endorsement_policy=f"{int(self.config.consensus_threshold * self.config.num_miners)}-of-{self.config.num_miners}"
        )
        
        # Setup Proof-of-Forecast consensus
        self.consensus_protocol = ProofOfForecastConsensus(
            threshold=self.config.consensus_threshold,
            max_debate_rounds=self.config.max_debate_rounds,
            blockchain_client=self.blockchain_client
        )
        
        self.logger.info("Blockchain setup completed")
    
    def setup_agents(self) -> Dict[str, Union[Publisher, Oracle, List[Miner], List[Evaluator], Leader]]:
        """Initialize multi-agent system"""
        self.logger.info("Setting up multi-agent system...")
        
        # Publisher Agent
        publisher = Publisher(
            agent_id="publisher_0",
            blockchain_client=self.blockchain_client,
            config=self.config
        )
        
        # Oracle Agent  
        oracle = Oracle(
            agent_id="oracle_0",
            blockchain_client=self.blockchain_client,
            update_frequency=self.config.oracle_update_frequency,
            config=self.config
        )
        
        # Miner Agents
        miners = []
        for i in range(self.config.num_miners):
            miner = Miner(
                agent_id=f"miner_{i}",
                blockchain_client=self.blockchain_client,
                device=self.device,
                config=self.config
            )
            miners.append(miner)
        
        # Evaluator Agents
        evaluators = []
        for i in range(self.config.num_evaluators):
            evaluator = Evaluator(
                agent_id=f"evaluator_{i}",
                blockchain_client=self.blockchain_client,
                config=self.config
            )
            evaluators.append(evaluator)
        
        # Leader Agent (initially None, elected dynamically)
        leader = Leader(
            agent_id="leader_0",
            blockchain_client=self.blockchain_client,
            config=self.config
        )
        
        self.agents = {
            'publisher': publisher,
            'oracle': oracle,
            'miners': miners,
            'evaluators': evaluators,
            'leader': leader
        }
        
        self.logger.info(f"Agent system setup completed: {self.config.num_miners} miners, {self.config.num_evaluators} evaluators")
        return self.agents
    
    def setup_global_model(self, model_config: Optional[Dict] = None) -> TaskAlignedLLM:
        """Initialize task-aligned global model"""
        self.logger.info("Setting up task-aligned global model...")
        
        # Initialize base LLM with task alignment (DEITA + DPO-TS)
        self.global_model = TaskAlignedLLM(
            model_name=self.config.model_name,
            model_path=self.config.model_path,
            qlora_config={
                'rank': self.config.qlora_rank,
                'alpha': self.config.qlora_alpha,
                'quantization_bits': self.config.quantization_bits
            },
            device=self.device,
            config=model_config or {}
        )
        
        self.logger.info(f"Global model initialized: {self.config.model_name}")
        return self.global_model
    
    def setup_incentives(self) -> Tuple[RewardSystem, StakeDynamics]:
        """Initialize reward system and stake dynamics"""
        self.logger.info("Setting up incentive mechanisms...")
        
        # Reward System
        self.reward_system = RewardSystem(
            performance_scale=self.config.performance_reward_scale,
            evaluation_scale=self.config.evaluation_reward_scale,
            oracle_scale=self.config.oracle_reward_scale,
            leader_scale=self.config.leader_reward_scale
        )
        
        # Stake Dynamics
        self.stake_dynamics = StakeDynamics(
            initial_stake=100.0,  # Default initial stake
            decay_rate=self.config.stake_decay_rate,
            slash_threshold=2.0  # Standard deviations for outlier detection
        )
        
        self.logger.info("Incentive mechanisms initialized")
        return self.reward_system, self.stake_dynamics
    
    def train(
        self,
        train_data: Dict[str, torch.utils.data.DataLoader],
        val_data: Optional[Dict[str, torch.utils.data.DataLoader]] = None,
        test_data: Optional[Dict[str, torch.utils.data.DataLoader]] = None
    ) -> Dict[str, any]:
        """
        Main training loop implementing BLEND federated learning
        
        Args:
            train_data: Training data for each miner
            val_data: Validation data (optional)
            test_data: Test data (optional)
            
        Returns:
            Training results and metrics
        """
        self.logger.info("Starting BLEND federated training...")
        
        # Initialize FedADAM optimizer
        fed_optimizer = FedADAMOptimizer(
            learning_rate=self.config.learning_rate,
            beta1=self.config.fedadam_beta1,
            beta2=self.config.fedadam_beta2
        )
        
        # Training loop
        for round_idx in range(self.config.global_rounds):
            self.current_round = round_idx
            self.logger.info(f"=== Round {round_idx + 1}/{self.config.global_rounds} ===")
            
            # Step 1: Task initialization and pool formation
            task_info = self.agents['publisher'].initialize_task(
                global_model=self.global_model,
                round_number=round_idx
            )
            
            # Step 2: Oracle information gathering
            context_data = self.agents['oracle'].gather_context()
            
            # Step 3: Global model deployment and broadcasting
            self.agents['publisher'].broadcast_model_and_context(
                model_state=self.global_model.state_dict(),
                context=context_data,
                task_info=task_info
            )
            
            # Step 4: Parallel miner training
            miner_proposals = {}
            for i, miner in enumerate(self.agents['miners']):
                if f"miner_{i}" in train_data:
                    # Local training
                    local_model, training_loss = miner.local_training(
                        global_model=self.global_model,
                        train_loader=train_data[f"miner_{i}"],
                        epochs=self.config.local_epochs,
                        context=context_data
                    )
                    
                    # Generate signed proposal
                    proposal = miner.generate_proposal(local_model, training_loss)
                    miner_proposals[miner.agent_id] = proposal
            
            # Step 5: Collaborative evaluation and scoring
            evaluation_scores = {}
            for evaluator in self.agents['evaluators']:
                scores = evaluator.evaluate_proposals(
                    proposals=miner_proposals,
                    validation_data=val_data
                )
                evaluation_scores[evaluator.agent_id] = scores
            
            # Step 6: Consensus and model selection
            consensus_result = self.consensus_protocol.reach_consensus(
                proposals=miner_proposals,
                evaluation_scores=evaluation_scores,
                current_round=round_idx
            )
            
            if consensus_result['consensus_reached']:
                # Step 7: Leader selection and model aggregation
                leader_id = consensus_result['elected_leader']
                best_proposal = consensus_result['selected_proposal']
                
                # FedADAM aggregation
                aggregated_model = fed_optimizer.aggregate_and_update(
                    global_model=self.global_model,
                    selected_update=best_proposal['model_update'],
                    round_number=round_idx
                )
                
                # Update global model
                self.global_model.load_state_dict(aggregated_model)
                
                # Step 8: Reward distribution
                rewards = self.reward_system.calculate_rewards(
                    proposals=miner_proposals,
                    evaluation_scores=evaluation_scores,
                    consensus_result=consensus_result
                )
                
                # Update stakes
                self.stake_dynamics.update_stakes(rewards, round_idx)
                
                # Step 9: Block creation and blockchain update
                block_data = {
                    'round': round_idx,
                    'model_hash': self._hash_model(self.global_model),
                    'consensus_result': consensus_result,
                    'rewards': rewards,
                    'timestamp': self._get_timestamp()
                }
                
                self.blockchain_client.create_block(block_data)
                
                # Log round results
                round_metrics = self._calculate_round_metrics(
                    proposals=miner_proposals,
                    consensus_result=consensus_result,
                    val_data=val_data
                )
                
                self.training_history.append(round_metrics)
                self.logger.info(f"Round {round_idx + 1} completed successfully")
                
            else:
                self.logger.warning(f"Consensus not reached in round {round_idx + 1}, retrying...")
                continue
        
        # Final evaluation
        final_results = self._final_evaluation(test_data)
        
        self.logger.info("BLEND training completed successfully")
        return {
            'training_history': self.training_history,
            'final_results': final_results,
            'global_model': self.global_model
        }
    
    def evaluate(
        self,
        test_data: Dict[str, torch.utils.data.DataLoader],
        prediction_horizons: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained BLEND model on test data
        
        Args:
            test_data: Test datasets
            prediction_horizons: Forecasting horizons to evaluate
            
        Returns:
            Evaluation results for each horizon and dataset
        """
        horizons = prediction_horizons or self.config.prediction_horizons
        results = {}
        
        self.global_model.eval()
        
        for dataset_name, data_loader in test_data.items():
            dataset_results = {}
            
            for horizon in horizons:
                mse_scores = []
                mae_scores = []
                
                with torch.no_grad():
                    for batch in data_loader:
                        inputs, targets = batch
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        predictions = self.global_model.forecast(
                            inputs, 
                            prediction_length=horizon
                        )
                        
                        # Calculate metrics
                        mse = torch.mean((predictions - targets) ** 2).item()
                        mae = torch.mean(torch.abs(predictions - targets)).item()
                        
                        mse_scores.append(mse)
                        mae_scores.append(mae)
                
                dataset_results[f'T{horizon}'] = {
                    'MSE': np.mean(mse_scores),
                    'MAE': np.mean(mae_scores)
                }
            
            results[dataset_name] = dataset_results
        
        return results
    
    def _hash_model(self, model: nn.Module) -> str:
        """Generate hash of model parameters for blockchain storage"""
        import hashlib
        
        # Concatenate all model parameters
        param_bytes = b''
        for param in model.parameters():
            param_bytes += param.data.cpu().numpy().tobytes()
        
        # Generate SHA-256 hash
        return hashlib.sha256(param_bytes).hexdigest()
    
    def _get_timestamp(self) -> int:
        """Get current timestamp for blockchain"""
        import time
        return int(time.time())
    
    def _calculate_round_metrics(
        self,
        proposals: Dict,
        consensus_result: Dict,
        val_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Calculate metrics for current training round"""
        metrics = {
            'round': self.current_round,
            'consensus_reached': consensus_result['consensus_reached'],
            'num_proposals': len(proposals),
            'selected_proposal_score': consensus_result.get('best_score', 0.0)
        }
        
        if val_data:
            # Add validation metrics
            val_results = self.evaluate(val_data, [96])  # Quick validation
            avg_mse = np.mean([
                results['T96']['MSE'] 
                for results in val_results.values()
            ])
            metrics['validation_mse'] = avg_mse
        
        return metrics
    
    def _final_evaluation(
        self, 
        test_data: Optional[Dict[str, torch.utils.data.DataLoader]]
    ) -> Dict[str, any]:
        """Perform final comprehensive evaluation"""
        if test_data is None:
            return {}
        
        # Comprehensive evaluation on all horizons
        final_results = self.evaluate(test_data, self.config.prediction_horizons)
        
        # Calculate aggregate statistics
        all_mse_scores = []
        all_mae_scores = []
        
        for dataset_results in final_results.values():
            for horizon_results in dataset_results.values():
                all_mse_scores.append(horizon_results['MSE'])
                all_mae_scores.append(horizon_results['MAE'])
        
        aggregate_stats = {
            'average_mse': np.mean(all_mse_scores),
            'average_mae': np.mean(all_mae_scores),
            'std_mse': np.std(all_mse_scores),
            'std_mae': np.std(all_mae_scores)
        }
        
        return {
            'detailed_results': final_results,
            'aggregate_stats': aggregate_stats,
            'training_rounds': self.current_round + 1,
            'consensus_success_rate': sum(
                1 for h in self.training_history 
                if h.get('consensus_reached', False)
            ) / len(self.training_history) if self.training_history else 0
        }
    
    def save_model(self, save_path: str) -> None:
        """Save trained BLEND model"""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'current_round': self.current_round
        }, save_path)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained BLEND model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        self.current_round = checkpoint.get('current_round', 0)
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_training_summary(self) -> Dict[str, any]:
        """Get comprehensive training summary"""
        if not self.training_history:
            return {"message": "No training history available"}
        
        # Calculate training statistics
        consensus_rates = [h.get('consensus_reached', False) for h in self.training_history]
        validation_mses = [h.get('validation_mse', 0) for h in self.training_history if 'validation_mse' in h]
        
        summary = {
            'total_rounds': len(self.training_history),
            'consensus_success_rate': sum(consensus_rates) / len(consensus_rates),
            'final_validation_mse': validation_mses[-1] if validation_mses else None,
            'best_validation_mse': min(validation_mses) if validation_mses else None,
            'convergence_round': self._find_convergence_round(),
            'training_stability': self._calculate_training_stability()
        }
        
        return summary
    
    def _find_convergence_round(self) -> Optional[int]:
        """Find round where model converged (validation loss stopped improving)"""
        if len(self.training_history) < 10:
            return None
        
        validation_mses = [
            h.get('validation_mse', float('inf')) 
            for h in self.training_history 
            if 'validation_mse' in h
        ]
        
        if len(validation_mses) < 10:
            return None
        
        # Find point where improvement rate drops below threshold
        for i in range(5, len(validation_mses) - 5):
            recent_improvement = (
                np.mean(validation_mses[i-5:i]) - 
                np.mean(validation_mses[i:i+5])
            ) / np.mean(validation_mses[i-5:i])
            
            if recent_improvement < 0.01:  # Less than 1% improvement
                return i
        
        return None
    
    def _calculate_training_stability(self) -> float:
        """Calculate training stability metric"""
        if len(self.training_history) < 5:
            return 1.0
        
        validation_mses = [
            h.get('validation_mse', 0) 
            for h in self.training_history 
            if 'validation_mse' in h
        ]
        
        if len(validation_mses) < 5:
            return 1.0
        
        # Calculate coefficient of variation for recent rounds
        recent_mses = validation_mses[-10:]
        cv = np.std(recent_mses) / np.mean(recent_mses) if np.mean(recent_mses) > 0 else 0
        
        # Stability score (lower CV = higher stability)
        stability = max(0, 1 - cv)
        return stability


# Utility functions for framework initialization
def create_blend_framework(
    config_path: str,
    device: Optional[torch.device] = None
) -> BLENDFramework:
    """
    Factory function to create BLEND framework from config file
    
    Args:
        config_path: Path to configuration YAML file
        device: PyTorch device
        
    Returns:
        Initialized BLEND framework
    """
    return BLENDFramework(config=config_path, device=device)


def setup_distributed_training(
    framework: BLENDFramework,
    data_config: Dict[str, any],
    blockchain_config: Optional[str] = None
) -> BLENDFramework:
    """
    Setup complete distributed training environment
    
    Args:
        framework: BLEND framework instance
        data_config: Data configuration dictionary
        blockchain_config: Optional blockchain configuration path
        
    Returns:
        Fully configured BLEND framework
    """
    # Setup blockchain infrastructure
    framework.setup_blockchain(blockchain_config)
    
    # Initialize multi-agent system
    framework.setup_agents()
    
    # Setup global model
    framework.setup_global_model()
    
    # Initialize incentive mechanisms
    framework.setup_incentives()
    
    return framework


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="BLEND Framework")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create and setup framework
    framework = create_blend_framework(args.config)
    framework = setup_distributed_training(
        framework, 
        data_config={"data_dir": args.data}
    )
    
    print("BLEND Framework ready for training!")