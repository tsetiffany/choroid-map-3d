import optuna
from optuna.trial import TrialState
import torch
import numpy as np
from train import train_model
import json
import os
from datetime import datetime


def objective(trial):
    """Objective function for Optuna optimization"""

    n_gpus = torch.cuda.device_count()
    gpu_id = trial.number % n_gpus
    
    # Suggest hyperparameters
    config = {
        # Architecture - these matter a lot
        'init_features': trial.suggest_categorical('init_features', [8, 16, 32]),
        'depth': trial.suggest_int('depth', 4, 7),
        
        # Training - very important
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [2, 4, 8]),
        
        # Augmentation - can be important for small datasets
        'use_vertical_flip': trial.suggest_categorical('use_vertical_flip', [True, False]),
        'use_elastic': trial.suggest_categorical('use_elastic', [True, False]),
        
        # Loss balance
        'loss_alpha': trial.suggest_float('loss_alpha', 0.2, 0.8),
        
        # Elastic parameters (always set, only used if use_elastic=True)
        'elastic_alpha': trial.suggest_int('elastic_alpha', 10, 30),
        'elastic_sigma': trial.suggest_int('elastic_sigma', 2, 5),
        
        # Fixed parameters from Choroidalyzer
        'max_features': 64,
        'up_type': 'conv_then_interpolate',
        'extra_out_conv': True,
        'max_epochs': 100,
        'early_stopping_patience': 15,
        'gpu_id': gpu_id,

    }
    
    # Create a unique directory for this trial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"optuna_trial_{trial.number}_{timestamp}"
    
    try:
        # Train model
        results = train_model(config)
        
        # Return the metric to optimize (negative because Optuna minimizes)
        # Using validation dice as the optimization target
        val_dice = results['best_val_metrics']['dice']
        
        # Report intermediate values for pruning
        trial.report(val_dice, step=results['best_epoch'])
        
        # Store test results as user attributes for later analysis
        trial.set_user_attr('test_dice', results['test_metrics']['dice'])
        trial.set_user_attr('test_iou', results['test_metrics']['iou'])
        trial.set_user_attr('best_epoch', results['best_epoch'])
        trial.set_user_attr('experiment_dir', exp_name)
        
        return val_dice  # Optuna will maximize this
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        return 0.0  # Return worst possible score


def run_optuna_search(n_trials=20, study_name=None):
    """Run Optuna hyperparameter optimization"""
    
    if study_name is None:
        study_name = f"vessel_unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create study directory
    study_dir = f"optuna_study_{study_name}"
    os.makedirs(study_dir, exist_ok=True)
    
    # Create Optuna study with:
    # - TPE sampler: Tree-structured Parzen Estimator (learns from previous trials)
    # - Median pruner: Stops unpromising trials early
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize dice score
        sampler=optuna.samplers.TPESampler(seed=42),  # Bayesian optimization
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage=f'sqlite:///{study_dir}/optuna_study.db',  # Save study for later
        load_if_exists=True  # Resume if interrupted
    )
    
    # Add callback to save intermediate results
    def save_callback(study, trial):
        # Save after each trial
        df = study.trials_dataframe()
        df.to_csv(os.path.join(study_dir, 'trials.csv'), index=False)
        
        # Save best trial info
        if study.best_trial:
            best_info = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'best_trial_number': study.best_trial.number,
                'test_dice': study.best_trial.user_attrs.get('test_dice', None),
                'test_iou': study.best_trial.user_attrs.get('test_iou', None),
            }
            with open(os.path.join(study_dir, 'best_trial.json'), 'w') as f:
                json.dump(best_info, f, indent=4)
    
    # Run optimization
    print(f"Starting Optuna optimization with {n_trials} trials")
    print(f"Study saved to: {study_dir}")
    print("Optuna will learn which hyperparameters matter and focus on promising regions\n")
    
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=4,
        callbacks=[save_callback],
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation dice: {study.best_value:.4f}")
    if 'test_dice' in study.best_trial.user_attrs:
        print(f"Best test dice: {study.best_trial.user_attrs['test_dice']:.4f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Analyze importance
    analyze_study(study, study_dir)
    
    return study


def analyze_study(study, study_dir):
    """Analyze Optuna study results"""
    
    # Get parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        
        print("\n" + "="*60)
        print("HYPERPARAMETER IMPORTANCE (learned by Optuna):")
        print("="*60)
        
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{param}: {imp:.3f}")
        
        # Save importance
        with open(os.path.join(study_dir, 'importance.json'), 'w') as f:
            json.dump(importance, f, indent=4)
            
    except Exception as e:
        print(f"Could not calculate importance: {e}")
    
    # Create visualization plots if possible
    try:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(study_dir, 'optimization_history.png'))
        plt.close()
        
        # Plot parameter importance
        if len(study.trials) > 5:
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(os.path.join(study_dir, 'param_importance.png'))
            plt.close()
        
        # Plot parallel coordinates of top trials
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(
            study, 
            params=[p for p in study.best_params.keys()]
        )
        plt.savefig(os.path.join(study_dir, 'parallel_coordinates.png'))
        plt.close()
        
        print(f"\nVisualization plots saved to {study_dir}/")
        
    except Exception as e:
        print(f"Could not create visualization plots: {e}")


def resume_study(study_name, additional_trials=10):
    """Resume a previous Optuna study"""
    
    study_dir = f"optuna_study_{study_name}"
    db_path = f'sqlite:///{study_dir}/optuna_study.db'
    
    # Load existing study
    study = optuna.load_study(
        study_name=study_name,
        storage=db_path
    )
    
    print(f"Resuming study: {study_name}")
    print(f"Completed trials: {len(study.trials)}")
    print(f"Best so far: {study.best_value:.4f}")
    print(f"Running {additional_trials} more trials...\n")
    
    # Continue optimization
    study.optimize(objective, n_trials=additional_trials)
    
    # Re-analyze
    analyze_study(study, study_dir)
    
    return study


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'run':
            # Run new study
            n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            run_optuna_search(n_trials)
            
        elif sys.argv[1] == 'resume':
            # Resume existing study
            if len(sys.argv) < 3:
                print("Usage: python optuna_search.py resume <study_name> [additional_trials]")
            else:
                study_name = sys.argv[2]
                additional_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 10
                resume_study(study_name, additional_trials)
                
        elif sys.argv[1] == 'analyze':
            # Just analyze existing study
            if len(sys.argv) < 3:
                print("Usage: python optuna_search.py analyze <study_name>")
            else:
                study_name = sys.argv[2]
                study_dir = f"optuna_study_{study_name}"
                study = optuna.load_study(
                    study_name=study_name,
                    storage=f'sqlite:///{study_dir}/optuna_study.db'
                )
                analyze_study(study, study_dir)
    else:
        print("Optuna Hyperparameter Optimization for Vessel Segmentation")
        print("\nUsage:")
        print("  python optuna_search.py run [n_trials]     # Start new optimization")
        print("  python optuna_search.py resume <study_name> [trials]  # Resume study")
        print("  python optuna_search.py analyze <study_name>  # Analyze existing study")
        print("\nOptuna uses Bayesian optimization to intelligently search hyperparameters")
        print("It learns which parameters matter and focuses on promising regions")