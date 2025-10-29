import itertools
import json
import os
from train import train_model
import pandas as pd


def generate_hyperparameter_configs():
    """Generate all hyperparameter configurations to test"""
    
    # Define hyperparameter search space
    hyperparams = {
        'init_features': [8, 16],  # Starting channels (Choroidalyzer uses 8)
        'depth': [5, 7],  # Model depth (Choroidalyzer uses 7)
        'learning_rate': [1e-3, 5e-4, 1e-4],
        'use_vertical_flip': [True, False],
        'use_elastic': [True, False],
        'loss_alpha': [0.3, 0.5, 0.7],  # BCE vs Dice loss weight
    }
    
    # Fixed parameters (matching Choroidalyzer)
    fixed_params = {
        'max_features': 64,  # Choroidalyzer caps at 64
        'up_type': 'conv_then_interpolate',  # Choroidalyzer's upsampling
        'extra_out_conv': True,  # Choroidalyzer uses extra output conv
        'batch_size': 4,
        'max_epochs': 100,
        'early_stopping_patience': 15,
        'elastic_alpha': 20,
        'elastic_sigma': 3
    }
    
    # Generate all combinations
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    
    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        config.update(fixed_params)
        configs.append(config)
    
    return configs


def run_hyperparameter_search():
    """Run hyperparameter search"""
    configs = generate_hyperparameter_configs()
    print(f"Total configurations to test: {len(configs)}")
    
    results_dir = "hyperparameter_results"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"Config: {json.dumps(config, indent=2)}")
        print(f"{'='*60}\n")
        
        try:
            results = train_model(config)
            
            # Add config to results
            results['config'] = config
            results['config_id'] = i
            all_results.append(results)
            
            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
            
            # Save as JSON too
            with open(os.path.join(results_dir, 'results.json'), 'w') as f:
                json.dump(all_results, f, indent=4)
                
        except Exception as e:
            print(f"Error in configuration {i}: {str(e)}")
            continue
    
    # Analyze results
    analyze_results(all_results, results_dir)
    
    return all_results


def analyze_results(results, results_dir):
    """Analyze hyperparameter tuning results"""
    df = pd.DataFrame(results)
    
    # Extract metrics
    for metric in ['dice', 'iou', 'accuracy']:
        df[f'test_{metric}'] = df['test_metrics'].apply(lambda x: x[metric])
        df[f'val_{metric}'] = df['best_val_metrics'].apply(lambda x: x[metric])
    
    # Sort by test dice score
    df_sorted = df.sort_values('test_dice', ascending=False)
    
    # Print top 5 configurations
    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS BY TEST DICE SCORE:")
    print("="*60)
    
    for i, row in df_sorted.head(5).iterrows():
        print(f"\nRank {i+1}:")
        print(f"Test Dice: {row['test_dice']:.4f}")
        print(f"Test IoU: {row['test_iou']:.4f}")
        print(f"Config: {json.dumps(row['config'], indent=2)}")
    
    # Save detailed analysis
    analysis = {
        'best_config': df_sorted.iloc[0]['config'],
        'best_test_dice': df_sorted.iloc[0]['test_dice'],
        'best_test_iou': df_sorted.iloc[0]['test_iou'],
        'hyperparameter_importance': {}
    }
    
    # Analyze impact of each hyperparameter
    for param in ['init_features', 'learning_rate', 'use_vertical_flip', 'use_elastic', 'loss_alpha']:
        param_analysis = []
        for value in df['config'].apply(lambda x: x[param]).unique():
            mask = df['config'].apply(lambda x: x[param]) == value
            mean_dice = df[mask]['test_dice'].mean()
            std_dice = df[mask]['test_dice'].std()
            param_analysis.append({
                'value': value,
                'mean_dice': mean_dice,
                'std_dice': std_dice,
                'count': mask.sum()
            })
        analysis['hyperparameter_importance'][param] = sorted(param_analysis, 
                                                            key=lambda x: x['mean_dice'], 
                                                            reverse=True)
    
    with open(os.path.join(results_dir, 'analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=4)
    
    # Create summary plots
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        params_to_plot = ['init_features', 'learning_rate', 'use_vertical_flip', 
                         'use_elastic', 'loss_alpha']
        
        for i, param in enumerate(params_to_plot):
            ax = axes[i]
            
            # Group by parameter value
            param_values = []
            mean_scores = []
            std_scores = []
            
            for item in analysis['hyperparameter_importance'][param]:
                param_values.append(str(item['value']))
                mean_scores.append(item['mean_dice'])
                std_scores.append(item['std_dice'])
            
            # Plot
            x = range(len(param_values))
            ax.bar(x, mean_scores, yerr=std_scores, capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(param_values, rotation=45 if param == 'learning_rate' else 0)
            ax.set_xlabel(param)
            ax.set_ylabel('Mean Test Dice')
            ax.set_title(f'Impact of {param}')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'hyperparameter_analysis.png'), dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Could not create plots: {e}")


if __name__ == "__main__":
    # For testing with fewer configurations
    test_mode = False
    
    if test_mode:
        # Test with just a few configurations
        print("Running in test mode with limited configurations...")
        test_config = {
            'init_features': 8,
            'depth': 7,
            'max_features': 64,
            'up_type': 'conv_then_interpolate',
            'extra_out_conv': True,
            'learning_rate': 1e-3,
            'batch_size': 4,
            'max_epochs': 5,  # Reduced for testing
            'early_stopping_patience': 3,
            'use_vertical_flip': True,
            'use_elastic': False,
            'elastic_alpha': 20,
            'elastic_sigma': 3,
            'loss_alpha': 0.5
        }
        results = train_model(test_config)
        print(f"\nTest run complete. Results: {results}")
    else:
        # Run full hyperparameter search
        run_hyperparameter_search()