import torch.utils.data as data
import os
import cv2
import numpy as np
from data import common

def default_loader(path):
    """Load image using OpenCV and handle both RGB and thermal images"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert BGR to RGB for color images
        return img[:, :, [2, 1, 0]]
    elif len(img.shape) == 2:
        # Grayscale/thermal image - add channel dimension
        return np.expand_dims(img, axis=2)
    else:
        return img

def npy_loader(path):
    """Load numpy array"""
    return np.load(path)

IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.npy',
]

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    """Recursively find all images in directory"""
    images = []
    if not os.path.isdir(dir):
        raise ValueError(f'{dir} is not a valid directory')

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return sorted(images)

def generate_lr_image(hr_image, scale):
    """Generate LR image from HR image using bicubic downsampling"""
    if len(hr_image.shape) == 2:
        hr_image = np.expand_dims(hr_image, axis=2)
    
    h, w = hr_image.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    
    if hr_image.shape[2] == 1:
        # For single channel (thermal) images
        lr_image = cv2.resize(hr_image[:,:,0], (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        lr_image = np.expand_dims(lr_image, axis=2)
    else:
        # For multi-channel images
        lr_image = cv2.resize(hr_image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    
    return lr_image


class ThermalDataset(data.Dataset):
    """
    Dataset specifically designed for thermal images
    Supports both paired (HR+LR) and unpaired (HR only) thermal data
    """
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.ext = getattr(self.opt, 'ext', '.png')
        self.train = True if self.opt.phase == 'train' else False
        
        # Get dataset paths
        self.hr_dir = getattr(self.opt, 'hr_dir', None)
        self.lr_dir = getattr(self.opt, 'lr_dir', None)
        
        if self.hr_dir is None:
            raise ValueError("hr_dir must be specified in options")
            
        # Load image lists
        self.images_hr = make_dataset(self.hr_dir)
        
        if self.lr_dir is not None:
            # Paired dataset (HR + LR)
            self.images_lr = make_dataset(self.lr_dir)
            if len(self.images_hr) != len(self.images_lr):
                print(f"Warning: HR images ({len(self.images_hr)}) and LR images ({len(self.images_lr)}) count mismatch")
                # Take minimum to avoid index errors
                min_len = min(len(self.images_hr), len(self.images_lr))
                self.images_hr = self.images_hr[:min_len]
                self.images_lr = self.images_lr[:min_len]
            self.paired_data = True
        else:
            # HR-only dataset (generate LR on-the-fly)
            self.images_lr = None
            self.paired_data = False
            
        print(f"Loaded {len(self.images_hr)} thermal images for {'training' if self.train else 'testing'}")
        
        # Calculate repeat factor for training
        if self.train:
            self.repeat = getattr(self.opt, 'test_every', 1000) // max(1, (len(self.images_hr) // getattr(self.opt, 'batch_size', 16)))
        else:
            self.repeat = 1

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        
        # Handle channel configuration for thermal images
        n_colors = getattr(self.opt, 'n_colors', 1)  # Default to 1 for thermal
        lr, hr = common.set_channel(lr, hr, n_channels=n_colors)
        
        # Convert to tensor
        rgb_range = getattr(self.opt, 'rgb_range', 1)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=rgb_range)
        
        return lr_tensor, hr_tensor

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        return idx % len(self.images_hr)

    def _get_patch(self, img_in, img_tar):
        """Extract patches for training or prepare images for testing"""
        patch_size = getattr(self.opt, 'patch_size', 192)
        scale = self.scale
        
        if self.train:
            # Training: extract random patches and apply augmentation
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)
        else:
            # Testing: ensure proper dimensions
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        
        return img_in, img_tar

    def _load_file(self, idx):
        """Load HR and LR thermal image pair"""
        idx = self._get_index(idx)
        
        # Load HR image
        if self.ext == '.npy':
            hr = npy_loader(self.images_hr[idx])
        else:
            hr = default_loader(self.images_hr[idx])
        
        # Ensure HR image has proper shape
        if len(hr.shape) == 2:
            hr = np.expand_dims(hr, axis=2)
        
        # Load or generate LR image
        if self.paired_data:
            # Load existing LR image
            if self.ext == '.npy':
                lr = npy_loader(self.images_lr[idx])
            else:
                lr = default_loader(self.images_lr[idx])
            
            if len(lr.shape) == 2:
                lr = np.expand_dims(lr, axis=2)
        else:
            # Generate LR from HR
            lr = generate_lr_image(hr, self.scale)
        
        return lr, hr

class CustomDataset(data.Dataset):
    """
    Generic custom dataset for backward compatibility
    """
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.ext = getattr(self.opt, 'ext', '.png')
        self.train = True if self.opt.phase == 'train' else False
        
        # Get dataset paths
        self.hr_dir = getattr(self.opt, 'hr_dir', None)
        self.lr_dir = getattr(self.opt, 'lr_dir', None)
        
        if self.hr_dir is None:
            raise ValueError("hr_dir must be specified in options")
            
        # Load image lists
        self.images_hr = make_dataset(self.hr_dir)
        
        if self.lr_dir is not None:
            self.images_lr = make_dataset(self.lr_dir)
            if len(self.images_hr) != len(self.images_lr):
                print(f"Warning: HR images ({len(self.images_hr)}) and LR images ({len(self.images_lr)}) count mismatch")
                min_len = min(len(self.images_hr), len(self.images_lr))
                self.images_hr = self.images_hr[:min_len]
                self.images_lr = self.images_lr[:min_len]
            self.paired_data = True
        else:
            self.images_lr = None
            self.paired_data = False
            
        print(f"Loaded {len(self.images_hr)} images for training")
        
        if self.train:
            self.repeat = getattr(self.opt, 'test_every', 1000) // max(1, (len(self.images_hr) // getattr(self.opt, 'batch_size', 16)))
        else:
            self.repeat = 1

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.opt.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.opt.rgb_range)
        return lr_tensor, hr_tensor

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        return idx % len(self.images_hr)

    def _get_patch(self, img_in, img_tar):
        patch_size = getattr(self.opt, 'patch_size', 192)
        scale = self.scale
        
        if self.train:
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        
        return img_in, img_tar

    def _load_file(self, idx):
        idx = self._get_index(idx)
        
        # Load HR image
        if self.ext == '.npy':
            hr = npy_loader(self.images_hr[idx])
        else:
            hr = default_loader(self.images_hr[idx])
        
        # Load or generate LR image
        if self.paired_data:
            if self.ext == '.npy':
                lr = npy_loader(self.images_lr[idx])
            else:
                lr = default_loader(self.images_lr[idx])
        else:
            lr = generate_lr_image(hr, self.scale)
        
        return lr, hr