"""
BLEND Multi-Agent System Implementation
Implements Publisher, Oracle, Miner, Evaluator, and Leader agents

Author: Raed Abdel-Sater
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import hashlib
import time
import logging
from dataclasses import dataclass
import requests
import json
from datetime import datetime

from ..models import TaskAlignedLLM, QLoRAAdapter
from ..blockchain import BlockchainClient
from ..utils import MetricsCalculator


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: float
    signature: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base class for all BLEND agents"""
    
    def __init__(
        self,
        agent_id: str,
        blockchain_client: Optional[BlockchainClient] = None,
        config: Optional[Dict] = None
    ):
        self.agent_id = agent_id
        self.blockchain_client = blockchain_client
        self.config = config or {}
        self.logger = logging.getLogger(f'BLEND.{self.__class__.__name__}.{agent_id}')
        
        # Agent state
        self.is_active = True
        self.stake = 100.0  # Initial stake
        self.reputation = 1.0
        self.message_history = []
        
        # Cryptographic keys (simplified)
        self.private_key = self._generate_private_key()
        self.public_key = self._generate_public_key()
    
    def _generate_private_key(self) -> str:
        """Generate private key for agent (simplified implementation)"""
        return hashlib.sha256(f"{self.agent_id}_{time.time()}".encode()).hexdigest()
    
    def _generate_public_key(self) -> str:
        """Generate public key from private key (simplified implementation)"""
        return hashlib.sha256(self.private_key.encode()).hexdigest()
    
    def sign_message(self, message: str) -> str:
        """Sign message with private key"""
        return hashlib.sha256(f"{message}_{self.private_key}".encode()).hexdigest()
    
    def verify_signature(self, message: str, signature: str, public_key: str) -> bool:
        """Verify message signature"""
        expected_signature = hashlib.sha256(f"{message}_{public_key}".encode()).hexdigest()
        return signature == expected_signature
    
    @abstractmethod
    def execute_role(self, **kwargs) -> Dict[str, Any]:
        """Execute agent's primary role"""
        pass
    
    def send_message(self, receiver_id: str, message_type: str, content: Dict) -> AgentMessage:
        """Send message to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            signature=self.sign_message(json.dumps(content, sort_keys=True))
        )
        
        self.message_history.append(message)
        return message
    
    def update_stake(self, delta: float) -> None:
        """Update agent's stake"""
        self.stake = max(0, self.stake + delta)
        self.logger.debug(f"Stake updated: {self.stake}")
    
    def update_reputation(self, performance_score: float) -> None:
        """Update agent's reputation based on performance"""
        alpha = 0.1  # Learning rate
        self.reputation = (1 - alpha) * self.reputation + alpha * performance_score
        self.logger.debug(f"Reputation updated: {self.reputation}")


class Publisher(BaseAgent):
    """
    Publisher Agent - Responsible for task initialization and global model deployment
    """
    
    def __init__(self, agent_id: str, blockchain_client: BlockchainClient, config: Dict):
        super().__init__(agent_id, blockchain_client, config)
        self.current_task = None
        self.registered_agents = {}
        
    def execute_role(self, **kwargs) -> Dict[str, Any]:
        """Execute publisher role - task initialization and broadcasting"""
        return self.initialize_task(**kwargs)
    
    def initialize_task(
        self,
        global_model: TaskAlignedLLM,
        round_number: int,
        task_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Initialize forecasting task and create agent pool"""
        
        self.logger.info(f"Initializing task for round {round_number}")
        
        # Create task definition
        task_info = {
            'task_id': f"forecast_task_{round_number}",
            'round_number': round_number,
            'model_hash': self._hash_model(global_model),
            'prediction_horizons': self.config.get('prediction_horizons', [96, 192, 336, 720]),
            'lookback_window': self.config.get('lookback_window', 720),
            'task_type': 'time_series_forecasting',
            'timestamp': time.time()
        }
        
        if task_config:
            task_info.update(task_config)
        
        self.current_task = task_info
        
        # Register task on blockchain
        if self.blockchain_client:
            self.blockchain_client.submit_transaction({
                'type': 'task_initialization',
                'publisher_id': self.agent_id,
                'task_info': task_info
            })
        
        return task_info
    
    def register_agent(self, agent_id: str, agent_type: str, public_key: str) -> bool:
        """Register an agent in the agent pool"""
        
        if agent_id not in self.registered_agents:
            self.registered_agents[agent_id] = {
                'agent_type': agent_type,
                'public_key': public_key,
                'registration_time': time.time(),
                'status': 'active'
            }
            
            self.logger.info(f"Registered agent: {agent_id} ({agent_type})")
            return True
        
        return False
    
    def broadcast_model_and_context(
        self,
        model_state: Dict[str, torch.Tensor],
        context: Dict[str, Any],
        task_info: Dict[str, Any]
    ) -> None:
        """Broadcast global model and context to all registered agents"""
        
        self.logger.info("Broadcasting global model and context")
        
        # Create broadcast message
        broadcast_content = {
            'model_hash': self._hash_state_dict(model_state),
            'context': context,
            'task_info': task_info,
            'round_number': task_info['round_number']
        }
        
        # Store model state (in practice, this would be distributed via IPFS or similar)
        model_storage_hash = self._store_model_state(model_state)
        broadcast_content['model_storage_hash'] = model_storage_hash
        
        # Broadcast to blockchain
        if self.blockchain_client:
            self.blockchain_client.submit_transaction({
                'type': 'model_broadcast',
                'publisher_id': self.agent_id,
                'content': broadcast_content,
                'timestamp': time.time()
            })
    
    def _hash_model(self, model: nn.Module) -> str:
        """Generate hash of model parameters"""
        param_bytes = b''
        for param in model.parameters():
            param_bytes += param.data.cpu().numpy().tobytes()
        return hashlib.sha256(param_bytes).hexdigest()
    
    def _hash_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Generate hash of model state dictionary"""
        param_bytes = b''
        for key in sorted(state_dict.keys()):
            param_bytes += state_dict[key].cpu().numpy().tobytes()
        return hashlib.sha256(param_bytes).hexdigest()
    
    def _store_model_state(self, model_state: Dict[str, torch.Tensor]) -> str:
        """Store model state and return storage hash (simplified implementation)"""
        # In practice, this would use IPFS or similar distributed storage
        storage_hash = self._hash_state_dict(model_state)
        return storage_hash


class Oracle(BaseAgent):
    """
    Oracle Agent - Context-aware data aggregation from external sources
    """
    
    def __init__(
        self,
        agent_id: str,
        blockchain_client: BlockchainClient,
        update_frequency: int = 5,
        config: Dict = None
    ):
        super().__init__(agent_id, blockchain_client, config)
        self.update_frequency = update_frequency
        self.context_sources = self._initialize_context_sources()
        self.context_cache = {}
        
    def execute_role(self, **kwargs) -> Dict[str, Any]:
        """Execute oracle role - gather and filter context data"""
        return self.gather_context(**kwargs)
    
    def _initialize_context_sources(self) -> Dict[str, str]:
        """Initialize external data source endpoints"""
        return {
            'weather_api': 'https://api.openweathermap.org/data/2.5',
            'traffic_api': 'https://api.traffic.com/v1',
            'news_api': 'https://newsapi.org/v2',
            'energy_api': 'https://api.eia.gov/v2'
        }
    
    def gather_context(self, forecast_horizon: int = 24) -> Dict[str, Any]:
        """Gather context-aware information for forecasting"""
        
        self.logger.info("Gathering context information")
        
        context_data = {
            'weather_forecast': self._fetch_weather_data(forecast_horizon),
            'traffic_conditions': self._fetch_traffic_data(),
            'news_events': self._fetch_relevant_news(),
            'energy_prices': self._fetch_energy_prices(),
            'timestamp': time.time(),
            'horizon_hours': forecast_horizon
        }
        
        # Filter and validate context
        filtered_context = self._filter_context(context_data)
        
        # Calculate information gain
        info_gain = self._calculate_information_gain(filtered_context)
        filtered_context['information_gain'] = info_gain
        
        # Cache context
        self.context_cache[time.time()] = filtered_context
        
        return filtered_context
    
    def _fetch_weather_data(self, hours: int) -> Dict[str, Any]:
        """Fetch weather forecast data (simplified implementation)"""
        try:
            # Simulated weather data - in practice, would call real API
            weather_data = {
                'temperature_forecast': np.random.normal(20, 5, hours).tolist(),
                'humidity_forecast': np.random.uniform(30, 80, hours).tolist(),
                'wind_forecast': np.random.uniform(0, 20, hours).tolist(),
                'precipitation_forecast': np.random.exponential(2, hours).tolist()
            }
            return weather_data
        except Exception as e:
            self.logger.warning(f"Failed to fetch weather data: {e}")
            return {}
    
    def _fetch_traffic_data(self) -> Dict[str, Any]:
        """Fetch traffic condition data"""
        try:
            # Simulated traffic data
            traffic_data = {
                'congestion_level': np.random.uniform(0, 1),
                'average_speed': np.random.uniform(20, 80),
                'incident_count': np.random.poisson(2),
                'peak_hours': [7, 8, 17, 18, 19]
            }
            return traffic_data
        except Exception as e:
            self.logger.warning(f"Failed to fetch traffic data: {e}")
            return {}
    
    def _fetch_relevant_news(self) -> Dict[str, Any]:
        """Fetch and filter relevant news events"""
        try:
            # Simulated news data - in practice, would use NLP to filter relevance
            news_events = {
                'energy_related_events': [
                    {'title': 'New renewable energy plant opens', 'relevance_score': 0.8},
                    {'title': 'Energy policy changes announced', 'relevance_score': 0.6}
                ],
                'traffic_related_events': [
                    {'title': 'Major highway construction begins', 'relevance_score': 0.7}
                ],
                'economic_events': [
                    {'title': 'Interest rates remain stable', 'relevance_score': 0.4}
                ]
            }
            return news_events
        except Exception as e:
            self.logger.warning(f"Failed to fetch news data: {e}")
            return {}
    
    def _fetch_energy_prices(self) -> Dict[str, Any]:
        """Fetch energy price data"""
        try:
            # Simulated energy price data
            price_data = {
                'current_price': np.random.uniform(0.08, 0.15),
                'price_trend': np.random.choice(['increasing', 'decreasing', 'stable']),
                'peak_price_hours': [18, 19, 20],
                'off_peak_multiplier': 0.7
            }
            return price_data
        except Exception as e:
            self.logger.warning(f"Failed to fetch energy price data: {e}")
            return {}
    
    def _filter_context(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter and clean context data using LLM"""
        
        # Simplified filtering - in practice would use LLM to assess relevance
        filtered_data = {}
        
        for key, value in context_data.items():
            if key == 'timestamp' or key == 'horizon_hours':
                filtered_data[key] = value
                continue
                
            if isinstance(value, dict) and value:  # Non-empty dict
                # Calculate relevance score
                relevance_score = self._calculate_relevance(key, value)
                if relevance_score > 0.3:  # Threshold for inclusion
                    filtered_data[key] = {
                        'data': value,
                        'relevance_score': relevance_score
                    }
        
        return filtered_data
    
    def _calculate_relevance(self, data_type: str, data: Dict) -> float:
        """Calculate relevance score for context data"""
        # Simplified relevance calculation
        base_relevance = {
            'weather_forecast': 0.8,
            'traffic_conditions': 0.6,
            'news_events': 0.4,
            'energy_prices': 0.9
        }
        
        return base_relevance.get(data_type, 0.5)
    
    def _calculate_information_gain(self, context_data: Dict[str, Any]) -> float:
        """Calculate information gain provided by context"""
        
        # Simplified information gain calculation
        total_gain = 0.0
        
        for key, value in context_data.items():
            if isinstance(value, dict) and 'relevance_score' in value:
                total_gain += value['relevance_score']
        
        # Normalize by number of context sources
        return total_gain / max(1, len([v for v in context_data.values() if isinstance(v, dict) and 'relevance_score' in v]))


class Miner(BaseAgent):
    """
    Miner Agent - Local model training and proposal generation
    """
    
    def __init__(
        self,
        agent_id: str,
        blockchain_client: BlockchainClient,
        device: torch.device,
        config: Dict
    ):
        super().__init__(agent_id, blockchain_client, config)
        self.device = device
        self.local_model = None
        self.local_data = None
        self.training_history = []
        
    def execute_role(self, **kwargs) -> Dict[str, Any]:
        """Execute miner role - local training and proposal generation"""
        return self.local_training(**kwargs)
    
    def local_training(
        self,
        global_model: TaskAlignedLLM,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 3,
        context: Optional[Dict] = None
    ) -> Tuple[TaskAlignedLLM, float]:
        """Perform local training on private data"""
        
        self.logger.info(f"Starting local training for {epochs} epochs")
        
        # Clone global model for local training
        self.local_model = self._clone_model(global_model)
        self.local_model.to(self.device)
        self.local_model.train()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.local_model.parameters(),
            lr=self.config.get('learning_rate', 1e-4)
        )
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Add context to inputs if available
                if context:
                    inputs = self._incorporate_context(inputs, context)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.local_model(inputs)
                
                # Calculate loss (MSE for time series forecasting)
                loss = nn.MSELoss()(predictions, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        avg_total_loss = total_loss / num_batches
        
        # Record training history
        training_record = {
            'round': self.config.get('current_round', 0),
            'epochs': epochs,
            'final_loss': avg_total_loss,
            'timestamp': time.time()
        }
        self.training_history.append(training_record)
        
        self.logger.info(f"Local training completed. Average loss: {avg_total_loss:.6f}")
        
        return self.local_model, avg_total_loss
    
    def generate_proposal(self, local_model: TaskAlignedLLM, training_loss: float) -> Dict[str, Any]:
        """Generate signed model proposal for consensus"""
        
        self.logger.info("Generating model proposal")
        
        # Calculate model update (difference from global model)
        model_update = self._calculate_model_update(local_model)
        
        # Create proposal
        proposal = {
            'miner_id': self.agent_id,
            'model_update': model_update,
            'training_loss': training_loss,
            'model_hash': self._hash_model(local_model),
            'timestamp': time.time(),
            'stake': self.stake,
            'reputation': self.reputation
        }
        
        # Sign proposal
        proposal_str = json.dumps(proposal, sort_keys=True, default=str)
        proposal['signature'] = self.sign_message(proposal_str)
        
        return proposal
    
    def _clone_model(self, model: TaskAlignedLLM) -> TaskAlignedLLM:
        """Create a deep copy of the model for local training"""
        # Simplified cloning - in practice would properly clone the model architecture
        cloned_model = TaskAlignedLLM(
            model_name=model.model_name,
            qlora_config=model.qlora_config,
            device=self.device
        )
        cloned_model.load_state_dict(model.state_dict())
        return cloned_model
    
    def _incorporate_context(self, inputs: torch.Tensor, context: Dict) -> torch.Tensor:
        """Incorporate context information into model inputs"""
        # Simplified context incorporation
        # In practice, would properly encode context as additional features
        return inputs
    
    def _calculate_model_update(self, local_model: TaskAlignedLLM) -> Dict[str, torch.Tensor]:
        """Calculate model parameter updates"""
        # Simplified - returns the full model state
        # In practice, would calculate the difference from global model
        return {name: param.cpu().clone() for name, param in local_model.named_parameters()}
    
    def _hash_model(self, model: nn.Module) -> str:
        """Generate hash of model parameters"""
        param_bytes = b''
        for param in model.parameters():
            param_bytes += param.data.cpu().numpy().tobytes()
        return hashlib.sha256(param_bytes).hexdigest()


class Evaluator(BaseAgent):
    """
    Evaluator Agent - Model validation and scoring
    """
    
    def __init__(self, agent_id: str, blockchain_client: BlockchainClient, config: Dict):
        super().__init__(agent_id, blockchain_client, config)
        self.evaluation_history = []
        self.metrics_calculator = MetricsCalculator()
        
    def execute_role(self, **kwargs) -> Dict[str, Any]:
        """Execute evaluator role - model proposal evaluation"""
        return self.evaluate_proposals(**kwargs)
    
    def evaluate_proposals(
        self,
        proposals: Dict[str, Dict],
        validation_data: Optional[Dict] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model proposals and assign scores"""
        
        self.logger.info(f"Evaluating {len(proposals)} proposals")
        
        evaluation_scores = {}
        
        for miner_id, proposal in proposals.items():
            try:
                # Verify proposal signature
                if not self._verify_proposal(proposal):
                    self.logger.warning(f"Invalid proposal signature from {miner_id}")
                    continue
                
                # Calculate performance score
                performance_score = self._calculate_performance_score(
                    proposal, validation_data
                )
                
                # Calculate additional scoring factors
                stake_weight = self._calculate_stake_weight(proposal.get('stake', 0))
                reputation_weight = self._calculate_reputation_weight(
                    proposal.get('reputation', 1.0)
                )
                
                # Combined score
                combined_score = (
                    0.7 * performance_score +
                    0.2 * stake_weight +
                    0.1 * reputation_weight
                )
                
                evaluation_scores[miner_id] = {
                    'performance_score': performance_score,
                    'stake_weight': stake_weight,
                    'reputation_weight': reputation_weight,
                    'combined_score': combined_score,
                    'evaluator_id': self.agent_id
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating proposal from {miner_id}: {e}")
                evaluation_scores[miner_id] = {
                    'error': str(e),
                    'combined_score': 0.0
                }
        
        # Record evaluation
        evaluation_record = {
            'timestamp': time.time(),
            'num_proposals': len(proposals),
            'scores': evaluation_scores
        }
        self.evaluation_history.append(evaluation_record)
        
        return evaluation_scores
    
    def _verify_proposal(self, proposal: Dict) -> bool:
        """Verify proposal signature and integrity"""
        if 'signature' not in proposal:
            return False
        
        # Extract signature and create proposal copy without signature
        signature = proposal.pop('signature')
        proposal_str = json.dumps(proposal, sort_keys=True, default=str)
        
        # Verify signature (simplified)
        expected_signature = hashlib.sha256(
            f"{proposal_str}_{proposal.get('miner_id', '')}".encode()
        ).hexdigest()
        
        # Restore signature to proposal
        proposal['signature'] = signature
        
        return signature == expected_signature
    
    def _calculate_performance_score(
        self,
        proposal: Dict,
        validation_data: Optional[Dict] = None
    ) -> float:
        """Calculate performance score based on training loss and validation"""
        
        # Base score from training loss (lower is better)
        training_loss = proposal.get('training_loss', float('inf'))
        if training_loss == float('inf'):
            return 0.0
        
        # Normalize training loss to score (0-1 range)
        # Using exponential decay: score = exp(-loss)
        base_score = np.exp(-training_loss)
        
        # If validation data available, calculate validation score
        if validation_data:
            # Simplified validation - in practice would load and test the model
            validation_bonus = 0.1  # Small bonus for having validation data
            base_score += validation_bonus
        
        return min(1.0, base_score)
    
    def _calculate_stake_weight(self, stake: float) -> float:
        """Calculate weight based on miner's stake"""
        # Logarithmic scaling to prevent stake dominance
        if stake <= 0:
            return 0.0
        
        return min(1.0, np.log(1 + stake) / np.log(1 + 1000))  # Normalize to max stake of 1000
    
    def _calculate_reputation_weight(self, reputation: float) -> float:
        """Calculate weight based on miner's reputation"""
        return min(1.0, max(0.0, reputation))


class Leader(BaseAgent):
    """
    Leader Agent - Consensus facilitation and block creation
    """
    
    def __init__(self, agent_id: str, blockchain_client: BlockchainClient, config: Dict):
        super().__init__(agent_id, blockchain_client, config)
        self.consensus_history = []
        
    def execute_role(self, **kwargs) -> Dict[str, Any]:
        """Execute leader role - facilitate consensus and create blocks"""
        return self.facilitate_consensus(**kwargs)
    
    def facilitate_consensus(
        self,
        proposals: Dict[str, Dict],
        evaluation_scores: Dict[str, Dict],
        round_number: int
    ) -> Dict[str, Any]:
        """Facilitate consensus among evaluators and select best proposal"""
        
        self.logger.info(f"Facilitating consensus for round {round_number}")
        
        # Aggregate evaluation scores
        aggregated_scores = self._aggregate_evaluation_scores(evaluation_scores)
        
        # Check for Byzantine fault tolerance
        consensus_threshold = self.config.get('consensus_threshold', 0.67)
        min_evaluators = max(1, int(len(evaluation_scores) * consensus_threshold))
        
        if len(evaluation_scores) < min_evaluators:
            return {
                'consensus_reached': False,
                'reason': 'Insufficient evaluators for consensus'
            }
        
        # Select best proposal
        best_proposal_id = self._select_best_proposal(aggregated_scores)
        
        if not best_proposal_id:
            return {
                'consensus_reached': False,
                'reason': 'No valid proposals found'
            }
        
        # Create consensus result
        consensus_result = {
            'consensus_reached': True,
            'elected_leader': self.agent_id,
            'selected_proposal': proposals[best_proposal_id],
            'best_proposal_id': best_proposal_id,
            'best_score': aggregated_scores.get(best_proposal_id, {}).get('combined_score', 0.0),
            'round_number': round_number,
            'timestamp': time.time(),
            'participating_evaluators': list(evaluation_scores.keys()),
            'consensus_threshold': consensus_threshold
        }
        
        # Record consensus
        self.consensus_history.append(consensus_result)
        
        return consensus_result
    
    def create_block(
        self,
        consensus_result: Dict,
        model_update: Dict[str, torch.Tensor],
        rewards: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create blockchain block with consensus results"""
        
        self.logger.info("Creating blockchain block")
        
        # Create block data
        block_data = {
            'block_type': 'consensus_block',
            'round_number': consensus_result['round_number'],
            'leader_id': self.agent_id,
            'selected_proposal_id': consensus_result['best_proposal_id'],
            'consensus_score': consensus_result['best_score'],
            'model_update_hash': self._hash_model_update(model_update),
            'rewards': rewards,
            'timestamp': time.time(),
            'previous_block_hash': self._get_previous_block_hash()
        }
        
        # Sign block
        block_str = json.dumps(block_data, sort_keys=True, default=str)
        block_data['block_signature'] = self.sign_message(block_str)
        
        # Submit to blockchain
        if self.blockchain_client:
            transaction_hash = self.blockchain_client.submit_transaction({
                'type': 'consensus_block',
                'block_data': block_data
            })
            block_data['transaction_hash'] = transaction_hash
        
        return block_data
    
    def _aggregate_evaluation_scores(
        self,
        evaluation_scores: Dict[str, Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate scores from multiple evaluators"""
        
        # Collect all miner IDs
        all_miners = set()
        for evaluator_scores in evaluation_scores.values():
            all_miners.update(evaluator_scores.keys())
        
        aggregated = {}
        
        for miner_id in all_miners:
            scores = []
            weights = []
            
            for evaluator_id, evaluator_scores in evaluation_scores.items():
                if miner_id in evaluator_scores and 'combined_score' in evaluator_scores[miner_id]:
                    score = evaluator_scores[miner_id]['combined_score']
                    # Weight by evaluator reputation (simplified)
                    weight = 1.0  # In practice, would use evaluator's reputation
                    
                    scores.append(score)
                    weights.append(weight)
            
            if scores:
                # Weighted average
                weighted_score = np.average(scores, weights=weights)
                aggregated[miner_id] = {
                    'combined_score': weighted_score,
                    'num_evaluators': len(scores),
                    'score_variance': np.var(scores) if len(scores) > 1 else 0.0
                }
        
        return aggregated
    
    def _select_best_proposal(self, aggregated_scores: Dict[str, Dict]) -> Optional[str]:
        """Select the best proposal based on aggregated scores"""
        
        if not aggregated_scores:
            return None
        
        # Find proposal with highest score
        best_proposal_id = None
        best_score = -1.0
        
        for miner_id, score_data in aggregated_scores.items():
            score = score_data.get('combined_score', 0.0)
            
            # Require minimum number of evaluators
            if score_data.get('num_evaluators', 0) < 1:
                continue
            
            if score > best_score:
                best_score = score
                best_proposal_id = miner_id
        
        return best_proposal_id
    
    def _hash_model_update(self, model_update: Dict[str, torch.Tensor]) -> str:
        """Generate hash of model update"""
        param_bytes = b''
        for key in sorted(model_update.keys()):
            param_bytes += model_update[key].cpu().numpy().tobytes()
        return hashlib.sha256(param_bytes).hexdigest()
    
    def _get_previous_block_hash(self) -> str:
        """Get hash of previous block from blockchain"""
        # Simplified implementation
        if self.blockchain_client:
            return self.blockchain_client.get_latest_block_hash()
        return "genesis_block"