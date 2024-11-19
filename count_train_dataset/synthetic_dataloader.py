import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from typing import Dict, Optional, Tuple
import torchvision.transforms as T
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPProcessor

class CLIPSyntheticDataset(Dataset):
    """Dataset class for loading synthetic image-caption pairs for CLIP fine-tuning"""
    
    def __init__(
        self,
        annotations_file: str,
        image_dir: str,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        """"Relies on clip preprocessing"""
        self.image_dir = image_dir
        
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.preprocess = CLIPProcessor.from_pretrained(model_name)

        self.max_length = self.preprocess.tokenizer.model_max_length

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get a single image-caption pair"""
        sample = self.annotations[idx]
        caption = sample['caption']

        image_path = os.path.join(self.image_dir, sample['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # enforce max length to avoid stacked tensor issues
        inputs = self.preprocess(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        return (inputs['pixel_values'].squeeze(0), inputs['input_ids'].squeeze(0))

def create_clip_dataloader(
    annotations_file: str,
    image_dir: str,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[DataLoader, Dataset]:
    """Create a DataLoader for CLIP fine-tuning"""
    
    dataset = CLIPSyntheticDataset(
        annotations_file=annotations_file,
        image_dir=image_dir,
        model_name=model_name
    )
    
    sampler = None
    if distributed:
        if not all([world_size, rank]):
            raise ValueError("world_size and rank required for distributed training")
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=(sampler is None)
    )
    
    return dataloader, dataset


if __name__ == "__main__":
    # Create dataloader
    annotations_file = "synthetic_dataset/synthetic_annotations.json"
    
    dataloader, dataset = create_clip_dataloader(
        annotations_file=annotations_file,
        batch_size=32,
        num_workers=1  # Reduced number of workers based on warning
    )
    
    # Example of iterating through batches
    for images, captions in dataloader:
        print(f"Images shape: {images.shape}")  # Shape: [