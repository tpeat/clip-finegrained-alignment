# rework to have true negative counts and image pad to square
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import json
import os
from typing import Dict, Optional, Tuple, List, Union
import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pad_image_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    max_dim = max(width, height)

    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    right = max_dim - width - left
    bottom = max_dim - height - top

    padded_image = Image.new(image.mode, (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (left, top))
    
    return padded_image

class CLIPSyntheticDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        image_dir: str,
        model_name: str = "ViT-B/32",
    ):
        self.image_dir = image_dir
        
        # TODO: optionally remove annotations with count 1 now
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
            
        _, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.preprocess = preprocess

        self.word_to_number = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        self.number_to_word = {v: k for k, v in self.word_to_number.items()}

    def __len__(self) -> int:
        return len(self.annotations)

    def create_negatives(self, caption: str) -> Tuple[int, str]:
        """Extract the count and object type from the caption's ending"""
        # Find the part after "with"
        index_after_with = caption.rindex("with")+5
        caption_prefix = caption[:index_after_with]
        count_phrase = caption[index_after_with:]  # +5 to skip "with "
        words = count_phrase.split()
        
        # First word should be the number
        count_word = words[0]
        gt_count = int(count_word) if count_word.isdigit() else self.word_to_number.get(count_word.lower(), 0)
        
        counterfactual_captions = []
        counts = []
        for count in list(set(range(1,11)) - set([gt_count])):
            counterfactual_caption = count_phrase.replace(str(gt_count), self.number_to_word[count])
            # add an s to the end of the object if we are making it plural
            if gt_count == 1 and counterfactual_caption[-1] != "s":
                counterfactual_caption += "s"
            counterfactual_captions.append(caption_prefix + counterfactual_caption)
            counts.append(count)
                
        return counterfactual_captions, gt_count, counts

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[str], List[int]]]:
        """Get an image with both template variants and count-based negatives"""
        sample = self.annotations[idx]
        original_caption = sample['caption']
        
        cf_captions, gt_count, cf_counts = self.create_negatives(original_caption)

        image_path = os.path.join(self.image_dir, sample['image_path'])
        image = Image.open(image_path).convert('RGB')
        padded_image = pad_image_to_square(image)
        
        image_tensor = self.preprocess(padded_image)
        
        text_tokens = clip.tokenize([original_caption]).squeeze(0)
        cf_tokens = clip.tokenize(cf_captions)

        cf_counts_tensor = torch.tensor(cf_counts, dtype=torch.int32)
        
        return {
            'image': image_tensor,
            'text': text_tokens,
            'cf_text': cf_tokens,
            'gt_count': gt_count,
            'cf_counts': cf_counts_tensor,
            'captions': cf_captions # for debugging
        }


def create_clip_dataloader(
    annotations_file: str,
    image_dir: str,
    model_name: str = "ViT-B/32",
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

