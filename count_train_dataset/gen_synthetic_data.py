import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm
import random
from typing import Tuple, List, Dict
import cv2
import shutil
from abc import ABC, abstractmethod

SIZE_CATEGORIES = {
    'extra_small' : (24,24),
    'small': (32, 96),
    'medium': (96, 224),
    'large': (224, 640) 
}

class COCOBaseDataset(ABC):
    """Abstract base class for COCO dataset handling"""
    def __init__(self, coco_dir="dataset/coco"):
        self.coco_dir = coco_dir
        self.train_ann_file = os.path.join(coco_dir, "annotations/instances_train2017.json")
        self.val_ann_file = os.path.join(coco_dir, "annotations/instances_val2017.json")
        self.captions_file = os.path.join(coco_dir, "annotations/captions_train2017.json")
        
        self.train_coco = COCO(self.train_ann_file)
        self.val_coco = COCO(self.val_ann_file)
        self.caption_coco = COCO(self.captions_file)

        # all possible CLIP categories
        self.categories = {category['id']: category['name'] for category in self.train_coco.loadCats(self.train_coco.getCatIds())}
        
        self.train_data = None
    
    def get_image_annotations(self, img_id, coco_instance):
        """Get all annotations for an image"""
        ann_ids = coco_instance.getAnnIds(imgIds=img_id)
        anns = coco_instance.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])
            
        return np.array(boxes), np.array(labels)
    
    def get_image_caption(self, img_id):
        """Get caption for an image"""
        ann_ids = self.caption_coco.getAnnIds(imgIds=img_id)
        anns = self.caption_coco.loadAnns(ann_ids)
        if anns:
            return anns[0]['caption']
        return ""

    def create_detection_dataset(self, split='train', output_dir='processed'):
        """Process COCO annotations into a detection dataset format"""
        os.makedirs(output_dir, exist_ok=True)
        
        coco_instance = self.train_coco if split == 'train' else self.val_coco
        img_dir = os.path.join(self.coco_dir, f"{split}2017")
        
        dataset = []
        for img_id in tqdm(coco_instance.getImgIds()):
            img_info = coco_instance.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            boxes, labels = self.get_image_annotations(img_id, coco_instance)
            
            # Skip images without annotations
            if len(boxes) == 0:
                continue
                
            dataset.append({
                'image_path': img_path,
                'boxes': boxes.tolist(),
                'labels': labels.tolist(),
                'width': img_info['width'],
                'height': img_info['height']
            })
        
        # Save dataset
        output_file = os.path.join(output_dir, f'{split}_detection.json')
        with open(output_file, 'w') as f:
            json.dump(dataset, f)
            
        if split == 'train':
            self.train_data = dataset
            
        return dataset

    @abstractmethod
    def create_dataset(self):
        """Abstract method for creating specialized datasets"""
        pass

    def visualize_sample(self, sample, show_labels=True, show_caption=True):
        """Base visualization method"""
        img = Image.open(sample['image_path'])
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        
        if 'boxes' in sample and 'labels' in sample:
            for box, label in zip(sample['boxes'], sample['labels']):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                if show_labels:
                    plt.text(x1, y1, self.categories[label],
                            bbox=dict(facecolor='white', alpha=0.7))
        
        if show_caption and 'caption' in sample:
            plt.figtext(0.5, 0.02, sample['caption'], 
                       wrap=True, horizontalalignment='center', 
                       fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.axis('off')
        plt.savefig('debug.png', bbox_inches='tight', pad_inches=0.5)


class COCOSyntheticDataset(COCOBaseDataset):
    def __init__(self, coco_dir="dataset/coco", output_dir="synthetic_dataset", clear_folder=True):
        super().__init__(coco_dir)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.captions_file = os.path.join(coco_dir, "annotations/captions_train2017.json")
        self.caption_coco = COCO(self.captions_file)

        # clear folder
        if clear_folder and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    def get_size_category(self, width: int, height: int) -> str:
        """Determine size category of object based on max dimension"""
        max_dim = max(width, height)
        if max_dim < SIZE_CATEGORIES['small'][1]:
            return 'small'
        elif max_dim < SIZE_CATEGORIES['medium'][1]:
            return 'medium'
        else:
            return 'large'
        
    def extract_object(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """Extract object using its bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))
    
    def random_placement_coords(self, obj_size: Tuple[int, int], 
                              target_size: Tuple[int, int]) -> Tuple[int, int]:
        """Generate random valid coordinates for object placement"""
        obj_w, obj_h = obj_size
        target_w, target_h = target_size
        
        x = random.randint(0, max(0, target_w - obj_w))
        y = random.randint(0, max(0, target_h - obj_h))
        
        return x, y

    def format_box_caption(self, boxes, label_name, image_width, image_height):
        """Format box coordinates into readable positions
        Don't love this method, not very accurate descriptions"""
        positions = []
        for box in boxes:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            
            # Determine horizontal position
            if x_center < image_width / 3:
                x_pos = "left"
            elif x_center < 2 * image_width / 3:
                x_pos = "center"
            else:
                x_pos = "right"
            
            # Determine vertical position
            if y_center < image_height / 3:
                y_pos = "top"
            elif y_center < 2 * image_height / 3:
                y_pos = "middle"
            else:
                y_pos = "bottom"
            
            positions.append(f"{y_pos}-{x_pos}")
        
        # Format positions string
        if len(positions) == 1:
            pos_str = positions[0]
        elif len(positions) == 2:
            pos_str = f"{positions[0]} and {positions[1]}"
        else:
            pos_str = ", ".join(positions[:-1]) + f", and {positions[-1]}"
        
        return f"{len(boxes)} {label_name}{'s' if len(boxes) > 1 else ''} at {pos_str}"
    
    def create_synthetic_dataset(self, num_samples: int, max_objects: int = 5, 
                           size_category: str = None, min_size: int = None, 
                           max_size: int = None, annotation_mode: str = 'full'):
        """Create synthetic dataset with size constraints and captions
        annotation_mode = ["full", "count" "integer"]"""
        if size_category and size_category not in SIZE_CATEGORIES:
            raise ValueError(f"Invalid size category. Choose from {SIZE_CATEGORIES.keys()}")
        
        if annotation_mode not in ['full', 'count', 'integer']:
            raise ValueError("annotation_mode must be one of: 'full', 'count', 'integer'")
        
        size_range = SIZE_CATEGORIES.get(size_category, None)
        min_size = min_size or (size_range[0] if size_range else 32)
        max_size = max_size or (size_range[1] if size_range else 640)
        
        synthetic_dataset = []

        metadata = {
            "num_samples" : num_samples,
            "max_objects" : max_objects,
            "size_category" : size_category,
            "min_size" : min_size,
            "annotation_mode" : annotation_mode,
            "using_count_captions" : annotation_mode== 'count'
        }
        
        with tqdm(total=num_samples) as pbar:
            while len(synthetic_dataset) < num_samples:
                src_sample = random.choice(self.train_data)
                dst_sample = random.choice(self.train_data)
                
                try:
                    if len(src_sample['boxes']) == 0:
                        continue
                        
                    # Check size before processing image
                    obj_idx = random.randint(0, len(src_sample['boxes']) - 1)
                    obj_bbox = src_sample['boxes'][obj_idx]
                    obj_width = obj_bbox[2] - obj_bbox[0]
                    obj_height = obj_bbox[3] - obj_bbox[1]
                    
                    if not (min_size <= max(obj_width, obj_height) <= max_size):
                        continue
                    
                    src_img = Image.open(src_sample['image_path'])
                    dst_img = Image.open(dst_sample['image_path']).convert('RGBA')
                    
                    obj_label = src_sample['labels'][obj_idx]
                    obj_img = self.extract_object(src_img, obj_bbox)
                    obj_img = obj_img.convert('RGBA')

                    # get og caption
                    dst_img_id = int(os.path.splitext(os.path.basename(dst_sample['image_path']))[0])
                    original_caption = self.get_image_caption(dst_img_id)
                    
                    # Decide number of times to place object
                    num_placements = random.randint(1, max_objects)
                    
                    # Create new image with annotations
                    new_boxes = []
                    new_labels = []
                    
                    # Place object multiple times
                    for _ in range(num_placements):
                        x, y = self.random_placement_coords(obj_img.size, dst_img.size)
                        
                        # Create new image for this placement
                        temp_img = dst_img.copy()
                        temp_img.paste(obj_img, (x, y), obj_img)
                        dst_img = temp_img
                        
                        # Record new bbox
                        new_box = [x, y, x + obj_img.width, y + obj_img.height]
                        new_boxes.append(new_box)
                        new_labels.append(obj_label)

                    obj_label_name = self.categories[obj_label]
                    
                    # Format caption based on annotation mode
                    if annotation_mode == 'count':
                        added_objects_caption = f"{len(new_boxes)} {obj_label_name}{'s' if len(new_boxes) > 1 else ''}"
                    elif annotation_mode == 'integer':
                        # Convert boxes to integer form: [x1, y1, x2, y2] -> number
                        box_integers = []
                        for box in new_boxes:
                            box_int = (int(box[0]) << 24) | (int(box[1]) << 16) | \
                                    (int(box[2]) << 8) | int(box[3])
                            box_integers.append(box_int)
                        added_objects_caption = f"{len(new_boxes)} {obj_label_name}{'s' if len(new_boxes) > 1 else ''} at positions {box_integers}"
                    else:  # 'full' mode
                        added_objects_caption = self.format_box_caption(
                            new_boxes, 
                            obj_label_name,
                            dst_img.width,
                            dst_img.height
                        )

                    full_caption = f"A photo of {original_caption} with {added_objects_caption}"
                    
                    # save to disk
                    output_path = os.path.join(self.output_dir, f"synthetic_{len(synthetic_dataset)}.png")
                    dst_img.convert('RGB').save(output_path)
  
                    annotation = {
                        'image_path': output_path,
                        'width': dst_img.width,
                        'height': dst_img.height,
                        'caption': full_caption,
                        'source_object': {
                            'image_path': src_sample['image_path'],
                            'bbox': obj_bbox,
                            'label': obj_label
                        }, 
                        'count': num_placements,
                        'boxes': new_boxes,
                        'labels' : new_labels
                    }
                    
                    # if annotation_mode != 'count':
                    #     annotation.update({
                    #         'boxes': new_boxes,
                    #         'labels': new_labels
                    #     })
                    if annotation_mode == 'integer':
                        annotation['box_integers'] = box_integers
                    
                    synthetic_dataset.append(annotation)
                        
                    pbar.update(1)

                    # failure recovery for scaling to 300k images -- saving intermediate json every 10k images
                    if (len(synthetic_dataset) % 10000 == 0):
                        intermediate_file = os.path.join(self.output_dir, f'synthetic_annotations_{idx + 1}.json')
                        with open(intermediate_file, 'w') as f:
                            json.dump({"metadata": metadata, "annotations": synthetic_dataset}, f)
                        print(f"Saved intermediate JSON: {intermediate_file}")

                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
        
        output_file = os.path.join(self.output_dir, 'synthetic_annotations.json')
        with open(output_file, 'w') as f:
            json.dump({"metadata": metadata, "annotations": synthetic_dataset}, f)
            
        return synthetic_dataset

    def create_dataset(self, num_samples: int, max_objects: int = 5, 
                      size_category: str = None, min_size: int = None, 
                      max_size: int = None, annotation_mode: str = 'full'):
        """Implementation of abstract method"""
        if self.train_data is None:
            self.train_data = self.create_detection_dataset('train')
        
        return self.create_synthetic_dataset(
            num_samples=num_samples,
            max_objects=max_objects,
            size_category=size_category,
            min_size=min_size,
            max_size=max_size,
            annotation_mode=annotation_mode
        )

    def visualize_sample(self, sample, show_labels=True, show_caption=True, show_integers=False):
        """
        Visualize an image with its annotations
        
        Args:
            show_integers (bool): If True, display integer representations of boxes
        """
        img = Image.open(sample['image_path'])
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        
        if 'boxes' in sample and 'labels' in sample:
            for i, (box, label) in enumerate(zip(sample['boxes'], sample['labels'])):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                if show_labels:
                    label_text = self.categories[label]
                    if show_integers and 'box_integers' in sample:
                        label_text += f"\n{sample['box_integers'][i]}"
                    plt.text(x1, y1, label_text,
                            bbox=dict(facecolor='white', alpha=0.7))
        
        if show_caption and 'caption' in sample:
            plt.figtext(0.5, 0.02, sample['caption'], 
                    wrap=True, horizontalalignment='center', 
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.axis('off')
        plt.savefig('debug.png', bbox_inches='tight', pad_inches=0.5)

def main():

    import argparse
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--max_objects', type=int, default=10,
                        help='Maximum number of object copies per image') 
    parser.add_argument('--size_category', type=str, default=None,
                        choices=['small', 'medium', 'large'],
                        help='Size category for sampled objects')
    parser.add_argument('--annotation_mode', type=str, default='full',
                        choices=['full', 'count', 'integer'],
                        help='Type of annotation to generate')
    parser.add_argument('--show_integers', action='store_true',
                        help='Show integer representations in visualization')
    parser.add_argument('--coco_dir', type=str, default="../dataset/coco",
                    help='Path to the COCO dataset directory')
    parser.add_argument('--output_dir', type=str, default="./synthetic_dataset",
                    help='Path to save the synthetic dataset')
    parser.add_argument('--max_obj', type=int, default=5,
                    help='Max Objects that be added to image')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                    help='Dataset split to process: train or val')


    args = parser.parse_args()


    coco_dir = args.coco_dir
    output_dir  = args.output_dir
    dataset = COCOSyntheticDataset(coco_dir=coco_dir,output_dir = output_dir)

    dataset.train_data = dataset.create_detection_dataset(args.split)

    synthetic_data = dataset.create_dataset(
        num_samples=args.num_samples,
        max_objects=args.max_objects,
        size_category=args.size_category,
        annotation_mode=args.annotation_mode
    )

    # TODO: add index control for which one we viz, with seeded randomization could be helpful to viz bad examples
    sample = random.choice(synthetic_data)
    dataset.visualize_sample(sample, show_integers=args.show_integers)

if __name__ == "__main__":
    main()