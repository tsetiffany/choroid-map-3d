import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import random

from dataset import OCTVesselDataset, Compose, RandomVerticalFlip, ElasticDeformation
from model import UNet, count_parameters


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        if self.pos_weight is None:
            # Calculate pos_weight dynamically based on batch
            num_pos = target.sum()
            num_neg = target.numel() - num_pos
            self.pos_weight = num_neg / (num_pos + 1e-6)
        
        return nn.functional.binary_cross_entropy_with_logits(
            pred, target, pos_weight=torch.tensor(self.pos_weight).to(pred.device)
        )


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.bce = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return self.alpha * self.bce(pred, target) + (1 - self.alpha) * self.dice(pred, target)


def calculate_metrics(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold
    target = target > 0.5
    
    # Calculate IoU
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Calculate Dice
    dice = (2 * intersection + 1e-6) / (pred.float().sum() + target.float().sum() + 1e-6)
    
    # Calculate pixel accuracy
    accuracy = (pred == target).float().mean()
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item()
    }


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def visualize_test_results(test_examples, save_dir, num_samples=5):
    """Visualize test results with overlay of predictions and ground truth"""
    
    # Flatten all examples
    all_images = []
    all_masks = []
    all_outputs = []
    
    for example in test_examples:
        for i in range(example['images'].shape[0]):
            all_images.append(example['images'][i])
            all_masks.append(example['masks'][i])
            all_outputs.append(example['outputs'][i])
    
    # Randomly select samples
    num_samples = min(num_samples, len(all_images))
    indices = random.sample(range(len(all_images)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, i in enumerate(indices):
        image = all_images[i][0].numpy()
        mask_true = all_masks[i][0].numpy()
        mask_pred = (all_outputs[i][0].numpy() > 0.5).astype(float)
        
        # Original image
        axes[idx, 0].imshow(image, cmap='gray')
        axes[idx, 0].set_title(f'Sample {i+1}: Input OCT')
        axes[idx, 0].axis('off')
        
        # Ground truth
        axes[idx, 1].imshow(mask_true, cmap='hot')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        # Prediction
        axes[idx, 2].imshow(mask_pred, cmap='hot')
        axes[idx, 2].set_title('Prediction')
        axes[idx, 2].axis('off')
        
        # Overlay
        overlay = np.zeros((*image.shape, 3))
        overlay[:, :, 0] = image  # Red channel: original image
        overlay[:, :, 1] = image  # Green channel: original image
        overlay[:, :, 2] = image  # Blue channel: original image
        
        # Normalize overlay
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
        
        # Add masks: Green for correct, Red for false positive, Blue for false negative
        tp = mask_true * mask_pred  # True positive
        fp = (1 - mask_true) * mask_pred  # False positive
        fn = mask_true * (1 - mask_pred)  # False negative
        
        overlay[:, :, 1] += tp * 0.5  # Green for true positive
        overlay[:, :, 0] += fp * 0.5  # Red for false positive
        overlay[:, :, 2] += fn * 0.5  # Blue for false negative
        
        overlay = np.clip(overlay, 0, 1)
        
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title('Overlay (G:TP, R:FP, B:FN)')
        axes[idx, 3].axis('off')
        
        # Calculate metrics for this sample
        dice = (2 * tp.sum() + 1e-6) / (mask_true.sum() + mask_pred.sum() + 1e-6)
        axes[idx, 3].text(0.02, 0.98, f'Dice: {dice:.3f}', 
                         transform=axes[idx, 3].transAxes, 
                         color='white', 
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_results_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved test visualization to {os.path.join(save_dir, 'test_results_visualization.png')}")


def train_model(config):
    # Configure GPU device if specified
    if 'gpu_id' in config:
        gpu_id = config['gpu_id']
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include GPU ID in experiment name if using multi-GPU
    gpu_suffix = f"_gpu{config['gpu_id']}" if 'gpu_id' in config else ""
    exp_name = f"unet_{config['init_features']}ch_lr{config['learning_rate']}_{'vflip' if config['use_vertical_flip'] else 'novflip'}_{'elastic' if config['use_elastic'] else 'noelastic'}{gpu_suffix}"
    exp_dir = os.path.join('experiments', f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set up data augmentation
    augmentations = []
    if config['use_vertical_flip']:
        augmentations.append(RandomVerticalFlip(p=0.5))
    if config['use_elastic']:
        augmentations.append(ElasticDeformation(
            alpha=config['elastic_alpha'], 
            sigma=config['elastic_sigma'], 
            p=0.5
        ))
    
    train_transform = Compose(augmentations) if augmentations else None
    
    # Create datasets
    data_dir = "CVI Training Data Set"
    train_indices = list(range(300))
    val_indices = list(range(300, 400))
    test_indices = list(range(400, 500))
    
    train_dataset = OCTVesselDataset(data_dir, transform=train_transform, indices=train_indices)
    val_dataset = OCTVesselDataset(data_dir, transform=None, indices=val_indices)
    test_dataset = OCTVesselDataset(data_dir, transform=None, indices=test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Create model
    model = UNet(
        in_channels=1,
        out_channels=1,
        depth=config['depth'],
        init_features=config['init_features'],
        max_features=config.get('max_features', 64),
        up_type=config.get('up_type', 'conv_then_interpolate'),
        extra_out_conv=config.get('extra_out_conv', True)
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    # Calculate class weights if not provided
    pos_weight = config.get('pos_weight', None)
    if pos_weight is None:
        # Calculate from training set
        print("Calculating class weights from training set...")
        total_pos = 0
        total_pixels = 0
        for batch in train_loader:
            masks = batch['mask']
            total_pos += masks.sum().item()
            total_pixels += masks.numel()
        pos_weight = (total_pixels - total_pos) / (total_pos + 1e-6)
        print(f"Calculated pos_weight: {pos_weight:.2f} (vessel pixels: {total_pos/total_pixels*100:.1f}%)")
    
    criterion = CombinedLoss(alpha=config['loss_alpha'], pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(exp_dir, 'tensorboard'))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['max_epochs']):
        # Training
        model.train()
        train_loss = 0
        train_metrics = {'iou': 0, 'dice': 0, 'accuracy': 0}
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["max_epochs"]} [Train]') as pbar:
            for batch in pbar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, masks)
                for k, v in metrics.items():
                    train_metrics[k] += v
                
                pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {'iou': 0, 'dice': 0, 'accuracy': 0}
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["max_epochs"]} [Val]') as pbar:
                for batch in pbar:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
                    
                    # Calculate metrics
                    metrics = calculate_metrics(outputs, masks)
                    for k, v in metrics.items():
                        val_metrics[k] += v
                    
                    pbar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/train_dice', train_metrics['dice'], epoch)
        writer.add_scalar('Metrics/val_dice', val_metrics['dice'], epoch)
        writer.add_scalar('Metrics/train_iou', train_metrics['iou'], epoch)
        writer.add_scalar('Metrics/val_iou', val_metrics['iou'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Dice: {train_metrics['dice']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(exp_dir, 'best_model.pth'))
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Test evaluation
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loss = 0
    test_metrics = {'iou': 0, 'dice': 0, 'accuracy': 0}
    
    # Store some examples for visualization
    test_examples = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Testing')):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            test_loss += loss.item()
            
            metrics = calculate_metrics(outputs, masks)
            for k, v in metrics.items():
                test_metrics[k] += v
            
            # Store first few batches for visualization
            if i < 2:  # Store first 2 batches (8 images with batch_size=4)
                test_examples.append({
                    'images': images.cpu(),
                    'masks': masks.cpu(),
                    'outputs': torch.sigmoid(outputs).cpu()
                })
    
    test_loss /= len(test_loader)
    for k in test_metrics:
        test_metrics[k] /= len(test_loader)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print(f"Test IoU: {test_metrics['iou']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Visualize test results
    visualize_test_results(test_examples, exp_dir)
    
    # Save raw heatmaps for dynamic thresholding
    print("\nSaving test heatmaps for dynamic thresholding...")
    all_heatmaps = []
    all_images = []
    all_masks = []
    
    for example in test_examples:
        for i in range(example['images'].shape[0]):
            all_heatmaps.append(example['outputs'][i, 0].numpy())
            all_images.append(example['images'][i, 0].numpy())
            all_masks.append(example['masks'][i, 0].numpy())
    
    np.savez_compressed(
        os.path.join(exp_dir, 'test_heatmaps.npz'),
        heatmaps=np.array(all_heatmaps),
        images=np.array(all_images),
        masks=np.array(all_masks)
    )
    print(f"Saved {len(all_heatmaps)} test heatmaps to {os.path.join(exp_dir, 'test_heatmaps.npz')}")
    
    # Save test results
    results = {
        'best_val_loss': best_val_loss,
        'best_val_metrics': checkpoint['val_metrics'],
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'best_epoch': checkpoint['epoch']
    }
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    writer.close()
    
    return results


if __name__ == "__main__":
    # Default configuration (matching Choroidalyzer baseline)
    config = {
        'init_features': 8,  # Choroidalyzer starts with 8
        'depth': 7,  # Choroidalyzer uses depth 7
        'max_features': 64,  # Choroidalyzer caps at 64
        'up_type': 'conv_then_interpolate',  # Choroidalyzer's upsampling
        'extra_out_conv': True,  # Choroidalyzer uses extra output conv
        'learning_rate': 1e-3,
        'batch_size': 4,
        'max_epochs': 500,  # Increased from 100 to 500
        'early_stopping_patience': 30,  # Increased patience too
        'use_vertical_flip': True,
        'use_elastic': False,
        'elastic_alpha': 20,
        'elastic_sigma': 3,
        'loss_alpha': 0.5  # Weight for BCE vs Dice loss
    }
    
    train_model(config)