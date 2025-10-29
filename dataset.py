import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


class OCTVesselDataset(Dataset):
    def __init__(self, data_dir, transform=None, indices=None):
        """
        Args:
            data_dir: Path to CVI Training Data Set directory
            transform: Optional transform to be applied on a sample
            indices: Optional list of indices to use (for train/val split)
        """
        self.data_dir = data_dir
        self.oct_dir = os.path.join(data_dir, 'oct')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.transform = transform
        
        # Get all files
        self.oct_files = sorted([f for f in os.listdir(self.oct_dir) if f.endswith('.mat')])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.mat')])
        
        # Use specific indices if provided
        if indices is not None:
            self.oct_files = [self.oct_files[i] for i in indices]
            self.mask_files = [self.mask_files[i] for i in indices]
        
        assert len(self.oct_files) == len(self.mask_files), "Number of OCT and mask files must match"
        
    def __len__(self):
        return len(self.oct_files)
    
    def __getitem__(self, idx):
        # Load OCT image
        oct_path = os.path.join(self.oct_dir, self.oct_files[idx])
        oct_mat = scipy.io.loadmat(oct_path)
        oct_image = oct_mat['image'].astype(np.float32)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask_mat = scipy.io.loadmat(mask_path)
        mask = mask_mat['mask'].astype(np.float32)
        
        # Normalize OCT image to [0, 1]
        oct_image = (oct_image - oct_image.min()) / (oct_image.max() - oct_image.min() + 1e-8)
        
        # Convert to torch tensors and add channel dimension
        oct_image = torch.from_numpy(oct_image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        sample = {'image': oct_image, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class RandomRotation:
    """Rotate both image and mask by the same random angle"""
    def __init__(self, degrees):
        self.degrees = degrees
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Random angle
        angle = np.random.uniform(-self.degrees, self.degrees)
        
        # Convert to PIL for rotation
        image_pil = transforms.ToPILImage()(image)
        mask_pil = transforms.ToPILImage()(mask)
        
        # Rotate
        image_pil = image_pil.rotate(angle, Image.BILINEAR)
        mask_pil = mask_pil.rotate(angle, Image.NEAREST)
        
        # Convert back to tensor
        image = transforms.ToTensor()(image_pil)
        mask = transforms.ToTensor()(mask_pil)
        
        return {'image': image, 'mask': mask}


class RandomHorizontalFlip:
    """Horizontally flip both image and mask with probability 0.5"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        if np.random.random() < self.p:
            image, mask = sample['image'], sample['mask']
            image = torch.flip(image, dims=[2])  # Flip along width dimension
            mask = torch.flip(mask, dims=[2])
            return {'image': image, 'mask': mask}
        return sample


class RandomVerticalFlip:
    """Vertically flip both image and mask with probability 0.5"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        if np.random.random() < self.p:
            image, mask = sample['image'], sample['mask']
            image = torch.flip(image, dims=[1])  # Flip along height dimension
            mask = torch.flip(mask, dims=[1])
            return {'image': image, 'mask': mask}
        return sample


class ElasticDeformation:
    """Apply elastic deformation to both image and mask"""
    def __init__(self, alpha=20, sigma=3, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, sample):
        if np.random.random() < self.p:
            image, mask = sample['image'], sample['mask']
            
            # Convert to numpy
            image_np = image.numpy().squeeze()
            mask_np = mask.numpy().squeeze()
            
            # Generate random displacement fields
            shape = image_np.shape
            dx = np.random.randn(*shape) * self.sigma
            dy = np.random.randn(*shape) * self.sigma
            
            # Smooth the displacement fields
            from scipy.ndimage import gaussian_filter
            dx = gaussian_filter(dx, self.sigma) * self.alpha
            dy = gaussian_filter(dy, self.sigma) * self.alpha
            
            # Create meshgrid
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            
            # Apply displacement
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            # Map coordinates
            from scipy.ndimage import map_coordinates
            image_deformed = map_coordinates(image_np, indices, order=1).reshape(shape)
            mask_deformed = map_coordinates(mask_np, indices, order=0).reshape(shape)
            
            # Convert back to tensor
            image = torch.from_numpy(image_deformed).unsqueeze(0).float()
            mask = torch.from_numpy(mask_deformed).unsqueeze(0).float()
            
            return {'image': image, 'mask': mask}
        return sample


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


def visualize_batch(dataloader, num_samples=4):
    """Visualize a batch of OCT images and masks"""
    batch = next(iter(dataloader))
    images = batch['image']
    masks = batch['mask']
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # OCT image
        axes[i, 0].imshow(images[i, 0].numpy(), cmap='gray')
        axes[i, 0].set_title(f'OCT Image {i+1}')
        axes[i, 0].axis('off')
        
        # Vessel mask
        axes[i, 1].imshow(masks[i, 0].numpy(), cmap='hot')
        axes[i, 1].set_title(f'Vessel Mask {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_batch.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to sample_batch.png")
    plt.close()


if __name__ == "__main__":
    # Define transforms
    train_transform = Compose([
        RandomRotation(degrees=10),
        RandomHorizontalFlip(p=0.5)
    ])
    
    # Create dataset
    data_dir = "CVI Training Data Set"
    
    # Use first 400 for training, last 100 for validation
    train_indices = list(range(400))
    val_indices = list(range(400, 500))
    
    train_dataset = OCTVesselDataset(data_dir, transform=train_transform, indices=train_indices)
    val_dataset = OCTVesselDataset(data_dir, transform=None, indices=val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Test the dataloader
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch mask shape: {batch['mask'].shape}")
    print(f"Image min/max: {batch['image'].min():.3f}/{batch['image'].max():.3f}")
    print(f"Mask unique values: {torch.unique(batch['mask'])}")
    
    # Visualize samples
    visualize_batch(train_loader, num_samples=4)