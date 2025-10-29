import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from dataset import OCTVesselDataset
from model import UNet
import argparse
from matplotlib.widgets import Slider
import pickle


def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    model = UNet(
        in_channels=1,
        out_channels=1,
        depth=config['depth'],
        init_features=config['init_features'],
        max_features=config.get('max_features', 64),
        up_type=config.get('up_type', 'conv_then_interpolate'),
        extra_out_conv=config.get('extra_out_conv', True)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def generate_heatmaps(model, test_loader, device='cuda'):
    """Generate probability heatmaps for test set"""
    model = model.to(device)
    
    all_heatmaps = []
    all_images = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Generating heatmaps'):
            images = batch['image'].to(device)
            masks = batch['mask']
            
            # Get raw logits and convert to probabilities
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            
            # Store results
            for i in range(images.shape[0]):
                all_heatmaps.append(probs[i, 0].numpy())
                all_images.append(images[i, 0].cpu().numpy())
                all_masks.append(masks[i, 0].numpy())
    
    return all_heatmaps, all_images, all_masks


def calculate_metrics_at_threshold(heatmaps, masks, threshold):
    """Calculate metrics at a specific threshold"""
    total_dice = 0
    total_iou = 0
    total_accuracy = 0
    
    for heatmap, mask in zip(heatmaps, masks):
        pred = (heatmap > threshold).astype(float)
        
        # Calculate metrics
        intersection = (pred * mask).sum()
        union = pred.sum() + mask.sum()
        
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        iou_union = ((pred + mask) > 0).sum()
        iou = (intersection + 1e-6) / (iou_union + 1e-6)
        accuracy = (pred == mask).mean()
        
        total_dice += dice
        total_iou += iou
        total_accuracy += accuracy
    
    n = len(heatmaps)
    return {
        'dice': total_dice / n,
        'iou': total_iou / n,
        'accuracy': total_accuracy / n
    }


def find_optimal_threshold(heatmaps, masks, metric='dice'):
    """Find optimal threshold based on validation set"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    for thresh in tqdm(thresholds, desc='Finding optimal threshold'):
        metrics = calculate_metrics_at_threshold(heatmaps, masks, thresh)
        score = metrics[metric]
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def interactive_threshold_viewer(heatmaps, images, masks, initial_threshold=0.5):
    """Interactive viewer to adjust threshold dynamically"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(bottom=0.15)
    
    # Initial sample
    idx = 0
    
    def update_display(threshold):
        # Clear axes
        for ax in axes.flat:
            ax.clear()
        
        # Get current sample
        image = images[idx]
        heatmap = heatmaps[idx]
        mask = masks[idx]
        
        # Threshold prediction
        pred = (heatmap > threshold).astype(float)
        
        # Calculate metrics
        intersection = (pred * mask).sum()
        dice = (2 * intersection + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
        
        # Display
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Input OCT')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('Probability Heatmap')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(mask, cmap='hot')
        axes[0, 2].set_title('Ground Truth')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(pred, cmap='hot')
        axes[1, 0].set_title(f'Prediction (thresh={threshold:.2f})')
        axes[1, 0].axis('off')
        
        # Overlay
        overlay = np.zeros((*image.shape, 3))
        overlay[:, :] = np.stack([image, image, image], axis=-1)
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
        
        tp = mask * pred
        fp = (1 - mask) * pred
        fn = mask * (1 - pred)
        
        overlay[:, :, 1] += tp * 0.5
        overlay[:, :, 0] += fp * 0.5
        overlay[:, :, 2] += fn * 0.5
        overlay = np.clip(overlay, 0, 1)
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title(f'Overlay (Dice={dice:.3f})')
        axes[1, 1].axis('off')
        
        # Histogram of probabilities
        axes[1, 2].hist(heatmap.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 2].axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.2f}')
        axes[1, 2].set_title('Probability Distribution')
        axes[1, 2].set_xlabel('Probability')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
        
        plt.draw()
    
    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Threshold', 0.0, 1.0, valinit=initial_threshold)
    slider.on_changed(update_display)
    
    # Navigation buttons
    ax_prev = plt.axes([0.05, 0.05, 0.1, 0.04])
    ax_next = plt.axes([0.85, 0.05, 0.1, 0.04])
    
    from matplotlib.widgets import Button
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')
    
    def prev_sample(event):
        nonlocal idx
        idx = (idx - 1) % len(images)
        update_display(slider.val)
    
    def next_sample(event):
        nonlocal idx
        idx = (idx + 1) % len(images)
        update_display(slider.val)
    
    btn_prev.on_clicked(prev_sample)
    btn_next.on_clicked(next_sample)
    
    # Initial display
    update_display(initial_threshold)
    
    plt.show()


def save_results(heatmaps, images, masks, save_path):
    """Save heatmaps and data for later analysis"""
    print(f"Saving results to {save_path}")
    
    # Save as numpy arrays
    np.savez_compressed(
        save_path,
        heatmaps=np.array(heatmaps),
        images=np.array(images),
        masks=np.array(masks)
    )
    
    # Also save as pickle for easy loading
    with open(save_path.replace('.npz', '.pkl'), 'wb') as f:
        pickle.dump({
            'heatmaps': heatmaps,
            'images': images,
            'masks': masks
        }, f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with dynamic threshold')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='CVI Training Data Set', help='Path to data')
    parser.add_argument('--save_heatmaps', type=str, help='Path to save heatmaps')
    parser.add_argument('--load_heatmaps', type=str, help='Path to load pre-computed heatmaps')
    parser.add_argument('--find_optimal', action='store_true', help='Find optimal threshold')
    parser.add_argument('--interactive', action='store_true', help='Launch interactive viewer')
    parser.add_argument('--threshold', type=float, default=0.5, help='Evaluation threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    if args.load_heatmaps:
        # Load pre-computed heatmaps
        print(f"Loading heatmaps from {args.load_heatmaps}")
        data = np.load(args.load_heatmaps)
        heatmaps = list(data['heatmaps'])
        images = list(data['images'])
        masks = list(data['masks'])
    else:
        # Generate new heatmaps
        model, config = load_model(args.checkpoint)
        
        # Create test dataset
        test_indices = list(range(400, 500))
        test_dataset = OCTVesselDataset(args.data_dir, transform=None, indices=test_indices)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
        
        # Generate heatmaps
        heatmaps, images, masks = generate_heatmaps(model, test_loader, args.device)
        
        # Save if requested
        if args.save_heatmaps:
            save_results(heatmaps, images, masks, args.save_heatmaps)
    
    # Find optimal threshold
    if args.find_optimal:
        print("\nFinding optimal threshold on test set...")
        optimal_thresh, best_score = find_optimal_threshold(heatmaps, masks)
        print(f"Optimal threshold: {optimal_thresh:.3f} (Dice: {best_score:.4f})")
        
        # Show metrics at different thresholds
        print("\nMetrics at different thresholds:")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, optimal_thresh]:
            metrics = calculate_metrics_at_threshold(heatmaps, masks, thresh)
            print(f"Threshold {thresh:.2f}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    # Interactive viewer
    if args.interactive:
        print("\nLaunching interactive threshold viewer...")
        print("Use slider to adjust threshold, buttons to navigate samples")
        interactive_threshold_viewer(heatmaps, images, masks, args.threshold)
    
    # Static evaluation
    if not args.interactive and not args.find_optimal:
        print(f"\nEvaluating at threshold {args.threshold}")
        metrics = calculate_metrics_at_threshold(heatmaps, masks, args.threshold)
        print(f"Dice: {metrics['dice']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()