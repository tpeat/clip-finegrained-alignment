import torch
from transformers import CLIPProcessor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions
from PIL import Image
from typing import Tuple, Dict, List
import os
import random

class COCOCLIPDataset(Dataset):
    def __init__(self, coco_dataset: CocoCaptions, model_name="openai/clip-vit-base-patch32", max_samples: int = 10000):
        """
        Creates a CLIP-compatible dataset from COCO Captions
        Args:
            coco_dataset: CocoCaptions dataset
            max_samples: Maximum number of samples to use (for faster training)
        """
        self.coco = coco_dataset
        # Take a subset of the dataset for faster training
        self.indices = list(range(len(self.coco)))
        random.shuffle(self.indices)
        self.indices = self.indices[:max_samples]
        
        # Get CLIP preprocessing
        self.preprocess = CLIPProcessor.from_pretrained(model_name)

        self.max_length = self.preprocess.tokenizer.model_max_length
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        true_idx = self.indices[idx]
        image, captions = self.coco[true_idx]
        
        # Randomly select one caption from the available captions
        caption = random.choice(captions)
        
        inputs = self.preprocess(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",  # Add this
            max_length=self.max_length,  # Add this
            truncation=True
        )
        
        return (
            # remove batching
            inputs.pixel_values[0],
            inputs.input_ids[0]
        )

def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)  # [B, C, H, W]
    texts = torch.stack(texts, dim=0)    # [B, max_length]
    return images, texts

def create_coco_dataloaders(
    root_dir: str,
    ann_file: str,
    batch_size: int = 32,
    model_name: str = "openai/clip-vit-base-patch32",
    max_samples: int = 10000
) -> DataLoader:
    """
    Creates DataLoader for COCO dataset
    """
    # Load COCO dataset
    coco_dataset = CocoCaptions(
        root=root_dir,
        annFile=ann_file,
        transform=None  # We'll use CLIP's preprocessing
    )
    
    # Create CLIP dataset
    clip_dataset = COCOCLIPDataset(coco_dataset, model_name=model_name, max_samples=max_samples)
    
    # Create dataloader
    dataloader = DataLoader(
        clip_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return dataloader