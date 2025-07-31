#!/usr/bin/env python3
"""
BLEND Training Script
Main script for training BLEND models with blockchain-enhanced federated learning

Author: Raed Abdel-Sater

Usage:
    python scripts/train_blend.py --config configs/default.yaml --dataset ETTh1
    python scripts/train_blend.py --config configs/ett.yaml --dataset all --output results/
"""

import os
import sys
import argparse
import json
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from blend import BLENDFramework, BLENDConfig
from blend.utils import DataLoader, MetricsCalculator, create_federated_splits
from blend.utils.visualization import plot_training_curves, plot_forecasting_results


def setup_logging(output_dir: str, verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('BLEND_Training')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    fh = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_config(config_path: str, overrides: Optional[Dict] = None) -> BLENDConfig:
    """Load and merge configuration"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply command line overrides
    if overrides:
        config_dict.update(overrides)
    
    return BLENDConfig(**config_dict)


def prepare_datasets(
    dataset_names: List[str],
    data_dir: str,
    num_miners: int,
    config: BLENDConfig,
    logger: logging.Logger
) -> Tuple[Dict, Dict, Dict]:
    """
    Prepare federated datasets for training
    
    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    train_data = {}
    val_data = {}
    test_data = {}
    
    for dataset_name in dataset_names:
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        data_loader = DataLoader(
            dataset_name=dataset_name,
            data_dir=data_dir,
            lookback_window=config.lookback_window,
            prediction_horizons=config.prediction_horizons,
            batch_size=config.batch_size
        )
        
        # Create federated splits
        fed_splits = create_federated_splits(
            dataset=data_loader.get_full_dataset(),
            num_clients=num_miners,
            split_strategy='iid',  # Can be 'iid' or 'non_iid'
            alpha=0.5  # Dirichlet parameter for non-IID
        )
        
        # Create data loaders for each miner
        for miner_id in range(num_miners):
            train_loader, val_loader, test_loader = data_loader.create_federated_loaders(
                client_data=fed_splits[miner_id],
                test_split=0.2,
                val_split=0.1
            )
            
            train_data[f"{dataset_name}_miner_{miner_id}"] = train_loader
            
            # Use first miner's data for global validation/testing
            if miner_id == 0:
                val_data[dataset_name] = val_loader
                test_data[dataset_name] = test_loader
    
    logger.info(f"Prepared federated datasets for {len(dataset_names)} datasets and {num_miners} miners")
    return train_data, val_data, test_data


def run_training_experiment(
    framework: BLENDFramework,
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    output_dir: str,
    logger: logging.Logger
) -> Dict:
    """Run complete training experiment"""
    
    logger.info("Starting BLEND training experiment...")
    start_time = datetime.now()
    
    # Train model
    training_results = framework.train(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data
    )
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    logger.info(f"Training completed in {training_duration:.2f} seconds")
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    final_results = framework.evaluate(
        test_data=test_data,
        prediction_horizons=framework.config.prediction_horizons
    )
    
    # Get training summary
    training_summary = framework.get_training_summary()
    
    # Combine all results
    experiment_results = {
        'training_results': training_results,
        'final_evaluation': final_results,
        'training_summary': training_summary,
        'training_duration': training_duration,
        'config': framework.config.__dict__,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_path = os.path.join(output_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    logger.info(f"Experiment results saved to {results_path}")
    
    return experiment_results


def generate_visualizations(
    results: Dict,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """Generate training and results visualizations"""
    
    logger.info("Generating visualizations...")
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Training curves
    if 'training_results' in results and 'training_history' in results['training_results']:
        plot_training_curves(
            training_history=results['training_results']['training_history'],
            save_path=os.path.join(viz_dir, 'training_curves.png')
        )
    
    # Performance comparison
    if 'final_evaluation' in results:
        plot_forecasting_results(
            results=results['final_evaluation'],
            save_path=os.path.join(viz_dir, 'performance_comparison.png')
        )
    
    logger.info(f"Visualizations saved to {viz_dir}")


def run_ablation_study(
    base_config: BLENDConfig,
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    output_dir: str,
    logger: logging.Logger
) -> Dict:
    """Run ablation study to evaluate component contributions"""
    
    logger.info("Starting ablation study...")
    
    ablation_results = {}
    
    # Ablation configurations
    ablation_configs = {
        'baseline': {
            'model_alignment': False,
            'oracle_agent': False,
            'proof_of_forecast': False
        },
        'with_alignment': {
            'model_alignment': True,
            'oracle_agent': False,
            'proof_of_forecast': False
        },
        'with_oracle': {
            'model_alignment': True,
            'oracle_agent': True, 
            'proof_of_forecast': False
        },
        'full_blend': {
            'model_alignment': True,
            'oracle_agent': True,
            'proof_of_forecast': True
        }
    }
    
    for config_name, config_overrides in ablation_configs.items():
        logger.info(f"Running ablation: {config_name}")
        
        # Create modified config
        ablation_config = BLENDConfig(**{
            **base_config.__dict__,
            **config_overrides
        })
        
        # Initialize framework
        framework = BLENDFramework(
            config=ablation_config,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Setup framework
        framework.setup_blockchain()
        framework.setup_agents()
        framework.setup_global_model()
        framework.setup_incentives()
        
        # Train and evaluate
        try:
            training_results = framework.train(
                train_data=train_data,
                val_data=val_data
            )
            
            evaluation_results = framework.evaluate(
                test_data=test_data,
                prediction_horizons=[96, 192, 336, 720]
            )
            
            ablation_results[config_name] = {
                'training_summary': framework.get_training_summary(),
                'evaluation_results': evaluation_results
            }
            
        except Exception as e:
            logger.error(f"Ablation {config_name} failed: {str(e)}")
            ablation_results[config_name] = {'error': str(e)}
    
    # Save ablation results
    ablation_path = os.path.join(output_dir, 'ablation_results.json')
    with open(ablation_path, 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    logger.info(f"Ablation study completed. Results saved to {ablation_path}")
    return ablation_results


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train BLEND model with blockchain-enhanced federated learning"
    )
    
    # Required arguments
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to configuration YAML file'
    )
    
    # Data arguments
    parser.add_argument(
        '--dataset', type=str, default='ETTh1',
        help='Dataset name or "all" for all datasets'
    )
    parser.add_argument(
        '--data_dir', type=str, default='./data',
        help='Data directory path'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of global training rounds (overrides config)'
    )
    parser.add_argument(
        '--miners', type=int, default=None,
        help='Number of miner agents (overrides config)'
    )
    parser.add_argument(
        '--local_epochs', type=int, default=None,
        help='Local training epochs per round'
    )
    
    # Output arguments
    parser.add_argument(
        '--output', type=str, default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--name', type=str, default=None,
        help='Experiment name (default: timestamp)'
    )
    
    # Execution options
    parser.add_argument(
        '--ablation', action='store_true',
        help='Run ablation study'
    )
    parser.add_argument(
        '--no_blockchain', action='store_true',
        help='Run without blockchain (for testing)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Training device (cuda/cpu/auto)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization plots'
    )
    
    args = parser.parse_args()
    
    # Setup experiment directory
    if args.name:
        exp_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"blend_experiment_{timestamp}"
    
    output_dir = os.path.join(args.output, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.verbose)
    logger.info(f"Starting BLEND experiment: {exp_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load configuration
    config_overrides = {}
    if args.epochs:
        config_overrides['global_rounds'] = args.epochs
    if args.miners:
        config_overrides['num_miners'] = args.miners
    if args.local_epochs:
        config_overrides['local_epochs'] = args.local_epochs
    
    config = load_config(args.config, config_overrides)
    logger.info(f"Configuration loaded: {config}")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Prepare datasets
    if args.dataset == 'all':
        dataset_names = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']
    else:
        dataset_names = [args.dataset]
    
    train_data, val_data, test_data = prepare_datasets(
        dataset_names=dataset_names,
        data_dir=args.data_dir,
        num_miners=config.num_miners,
        config=config,
        logger=logger
    )
    
    # Initialize BLEND framework
    logger.info("Initializing BLEND framework...")
    framework = BLENDFramework(config=config, device=device)
    
    # Setup framework components
    if not args.no_blockchain:
        framework.setup_blockchain()
    
    framework.setup_agents()
    framework.setup_global_model()
    framework.setup_incentives()
    
    try:
        if args.ablation:
            # Run ablation study
            results = run_ablation_study(
                base_config=config,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                output_dir=output_dir,
                logger=logger
            )
        else:
            # Run main training experiment
            results = run_training_experiment(
                framework=framework,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                output_dir=output_dir,
                logger=logger
            )
            
            # Save trained model
            model_path = os.path.join(output_dir, 'trained_model.pth')
            framework.save_model(model_path)
            logger.info(f"Trained model saved to {model_path}")
        
        # Generate visualizations if requested
        if args.visualize:
            generate_visualizations(results, output_dir, logger)
        
        logger.info("Experiment completed successfully!")
        
        # Print summary
        if not args.ablation and 'training_summary' in results:
            summary = results['training_summary']
            print(f"\n=== Training Summary ===")
            print(f"Total rounds: {summary.get('total_rounds', 'N/A')}")
            print(f"Consensus success rate: {summary.get('consensus_success_rate', 0):.2%}")
            print(f"Final validation MSE: {summary.get('final_validation_mse', 'N/A')}")
            print(f"Training stability: {summary.get('training_stability', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()