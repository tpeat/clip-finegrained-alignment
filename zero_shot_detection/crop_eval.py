import torch
import clip
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import json
from typing import Tuple, Dict, List

def create_white_square_image(image_size=(224, 224), bbox=None):
    """Plain white image"""
    img = Image.new('RGB', image_size, color='white')
    # NOTE: if you add a white square to this, then its flips is guess to positive
    return img

def save_image_with_bbox(image, bbox, save_path, title=""):
    """Helper function to save image with bounding box visualization"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    rect = patches.Rectangle(
        (bbox[0], bbox[1]), bbox[2], bbox[3],
        linewidth=2, edgecolor='r', facecolor='none'
    )
    plt.gca().add_patch(rect)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_box_area_ratio(bbox, img_width, img_height):
    """Calculate the ratio of bbox area to image area"""
    box_width = bbox[2]
    box_height = bbox[3]
    box_area = box_width * box_height
    img_area = img_width * img_height
    return box_area / img_area

def find_small_object_image(coco_dataset):
    """Find a random image with exactly one small object instance of its category"""
    while True:
        img_id = random.choice(list(coco_dataset.train_coco.imgs.keys()))
        img_info = coco_dataset.train_coco.imgs[img_id]
        ann_ids = coco_dataset.train_coco.getAnnIds(imgIds=img_id)
        anns = coco_dataset.train_coco.loadAnns(ann_ids)
        
        category_counts = {}
        small_object_ann = None
        
        for ann in anns:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
            
            ratio = get_box_area_ratio(ann['bbox'], img_info['width'], img_info['height'])
            if ratio < 0.005:  # Object with <0.5% area
                if small_object_ann is None:
                    small_object_ann = ann
                    small_object_category = cat_id
        
        # ensure its the only instance in the image
        if (small_object_ann is not None and 
            category_counts[small_object_category] == 1):
            return img_id, small_object_ann


def crop_to_target_ratio(img, bbox, target_ratio):
    """Crop image so bbox takes up target_ratio of area"""
    img_width, img_height = img.size
    box_x, box_y, box_width, box_height = bbox
    box_area = box_width * box_height
    
    target_area = box_area / target_ratio
    scale_factor = np.sqrt(target_area / (img_width * img_height))
    
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)
    
    center_x = box_x + box_width/2
    center_y = box_y + box_height/2
    
    crop_x1 = max(0, int(center_x - new_width/2))
    crop_y1 = max(0, int(center_y - new_height/2))
    crop_x2 = min(img_width, crop_x1 + new_width)
    crop_y2 = min(img_height, crop_y1 + new_height)
    
    cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    new_bbox = [
        box_x - crop_x1,
        box_y - crop_y1,
        box_width,
        box_height
    ]
    
    return cropped_img, new_bbox

class CLIPEvaluator:
    def __init__(self, coco_dir: str = "dataset/coco", use_white_square: bool = False, debug: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.coco_dir = coco_dir
        self.train_ann_file = os.path.join(coco_dir, "annotations/instances_train2017.json")
        self.train_coco = COCO(self.train_ann_file)
        self.categories = {cat['id']: cat['name'] for cat in self.train_coco.loadCats(self.train_coco.getCatIds())}

        self.use_white_square = use_white_square
        self.debug = debug
        
    def load_image(self, img_id: int) -> Tuple[Image.Image, dict]:
        """Load image given image ID"""
        img_info = self.train_coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.coco_dir, 'train2017', img_info['file_name'])
        return Image.open(img_path).convert('RGB'), img_info

    def get_random_different_category(self, img_id: int, current_category_id: int) -> str:
        """Get a random category that's not present anywhere in the image"""
        # Get all annotations for this image
        ann_ids = self.train_coco.getAnnIds(imgIds=img_id)
        anns = self.train_coco.loadAnns(ann_ids)
        
        # Get all category IDs present in this image
        present_categories = {ann['category_id'] for ann in anns}
        
        # Filter out all categories that are present in the image
        all_categories = list(self.categories.items())
        filtered_categories = [(id, name) for id, name in all_categories 
                            if id not in present_categories]
        
        if not filtered_categories:
            raise ValueError("No suitable negative categories found for this image")
            
        random_id, random_name = random.choice(filtered_categories)
        return random_name
    
    def get_clip_score(self, image: Image.Image, object_name: str) -> Tuple[float, float]:
        """Get CLIP scores for presence and absence of object"""
        # TODO: potentially pad it, view what happens after preprocessing
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        text_prompts = [
            f"A photo with {object_name}",
            f"A photo with no {object_name}"
        ]
        print(text_prompts)
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
        return similarity[0][0].item(), similarity[0][1].item()
    
    def evaluate_single_image(self, img_id: int, annotation: dict) -> Dict[str, dict]:
        """Evaluate CLIP scores for an image at different crop ratios"""

        if self.use_white_square:
            image_size = (224, 224)
            image = create_white_square_image(image_size)
            bbox = [50, 50, 50, 50]
        else:
            image, img_info = self.load_image(img_id)
            bbox = annotation['bbox']

        true_object_name = self.categories[annotation['category_id']]
        false_object_name = self.get_random_different_category(img_id, annotation['category_id'])
        
        if self.debug:
            print("True object:", true_object_name)
            print("False object:", false_object_name)

            save_dir = f"evaluation_images/{img_id}_{true_object_name}"
            os.makedirs(save_dir, exist_ok=True)

        results = {}
        
        original_pos_true, original_neg_true = self.get_clip_score(image, true_object_name)
        results['original_positive'] = {
            'object_name': true_object_name,
            'positive_score': original_pos_true,
            'negative_score': original_neg_true,
            'correct': original_pos_true > original_neg_true,
            'ground_truth': 'positive'
        }

        original_pos_false, original_neg_false = self.get_clip_score(image, false_object_name)
        results['original_negative'] = {
            'object_name': false_object_name,
            'positive_score': original_pos_false,
            'negative_score': original_neg_false,
            'correct': original_neg_false > original_pos_false,  # Note reversed condition
            'ground_truth': 'negative'
        }

        if self.debug:
            save_image_with_bbox(
                image, bbox, 
                f"{save_dir}/original_positive.png",
                f"Original - True {true_object_name} ({original_pos_true:.2f} vs {original_neg_true:.2f})"
            )
        
        crop_ratios = [0.05, 0.1]
        for ratio in crop_ratios:
            cropped_img, new_bbox = crop_to_target_ratio(image, bbox, ratio)

            pos_score_true, neg_score_true = self.get_clip_score(cropped_img, true_object_name)
            results[f'crop_{int(ratio*100):02d}_positive'] = {
                'object_name': true_object_name,
                'positive_score': pos_score_true,
                'negative_score': neg_score_true,
                'correct': pos_score_true > neg_score_true,
                'ground_truth': 'positive'
            }

            if self.debug:
                save_image_with_bbox(
                    cropped_img, new_bbox,
                    f"{save_dir}/crop_{int(ratio*100)}_positive.png",
                    f"{int(ratio*100)}% Crop - True {true_object_name} ({pos_score_true:.2f} vs {neg_score_true:.2f})"
                )

            pos_score_false, neg_score_false = self.get_clip_score(cropped_img, false_object_name)
            results[f'crop_{int(ratio*100):02d}_negative'] = {
                'object_name': false_object_name,
                'positive_score': pos_score_false,
                'negative_score': neg_score_false,
                'correct': neg_score_false > pos_score_false,  # Note reversed condition
                'ground_truth': 'negative'
            }
                
        return results
    
    def run_evaluation(self, num_samples: int = 100) -> Dict[str, dict]:
        """Run evaluation on multiple random images"""
        all_results = []
        
        while len(all_results) < num_samples:
            try:
                img_id, ann = find_small_object_image(self)
                results = self.evaluate_single_image(img_id, ann)
                results['image_id'] = img_id
                results['category'] = self.categories[ann['category_id']]
                all_results.append(results)
                
                if len(all_results) % 10 == 0:
                    print(f"Processed {len(all_results)} images")
                    
            except Exception as e:
                print(f"Error processing image {img_id}: {str(e)}")
                continue

        aggregated_stats = self._aggregate_results(all_results)
        
        return {
            'individual_results': all_results,
            'aggregate_stats': aggregated_stats
        }
    
    def _aggregate_results(self, results: List[dict]) -> dict:
        """Aggregate results across all evaluated images, separating positive and negative samples"""
        stats = {
            'original_positive': {'correct': 0, 'avg_positive': 0, 'avg_negative': 0},
            'original_negative': {'correct': 0, 'avg_positive': 0, 'avg_negative': 0},
            'crop_05_positive': {'correct': 0, 'avg_positive': 0, 'avg_negative': 0},
            'crop_05_negative': {'correct': 0, 'avg_positive': 0, 'avg_negative': 0},
            'crop_10_positive': {'correct': 0, 'avg_positive': 0, 'avg_negative': 0},
            'crop_10_negative': {'correct': 0, 'avg_positive': 0, 'avg_negative': 0}
        }
        
        n = len(results)
        for result in results:
            for key in stats.keys():
                stats[key]['correct'] += int(result[key]['correct'])
                stats[key]['avg_positive'] += result[key]['positive_score']
                stats[key]['avg_negative'] += result[key]['negative_score']
        
        for key in stats:
            stats[key]['accuracy'] = stats[key]['correct'] / n
            stats[key]['avg_positive'] /= n
            stats[key]['avg_negative'] /= n
            
        return stats

def main():

    os.makedirs("evaluation_images", exist_ok=True)

    # white square is a sanity check to make sure its paying attention to visuals
    use_white_square = False
    debug = False
    evaluator = CLIPEvaluator("../dataset/coco", use_white_square=use_white_square, debug=debug)
    
    num_samples = 500 # TODO: scale this
    results = evaluator.run_evaluation(num_samples=num_samples)

    with open('clip_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation Summary:")
    for crop_type, stats in results['aggregate_stats'].items():
        print(f"\n{crop_type.capitalize()} Results:")
        print(f"Accuracy: {stats['accuracy']:.2%}")
        print(f"Average Positive Score: {stats['avg_positive']:.3f}")
        print(f"Average Negative Score: {stats['avg_negative']:.3f}")

if __name__ == "__main__":
    main()