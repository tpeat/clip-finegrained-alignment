"""
Extending Zero Shot CLIP Detection Approach from 
https://github.com/deepmancer/clip-object-detection/tree/main

And inspired from Facebook Detectron's pascal_voc_evaluation.py

"""
import requests
from pascal_data import label_prompts, pascal_classes
from io import BytesIO
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional
import torchvision
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
import clip
import argparse
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as F
from tqdm import tqdm
import os
from collections import defaultdict
import detectron2

class PascalVOCDetectionEvaluator:
    def __init__(self, annotations, class_names):
        self.annotations = annotations
        self.class_names = class_names

    def evaluate(self, predictions, iou_threshold=0.5):
        aps = {}
        for class_name in self.class_names:
            gt_boxes = {img_id: [obj["bbox"] for obj in objs if obj["name"] == class_name] 
                        for img_id, objs in self.annotations.items()}
            pred_boxes = predictions.get(class_name, [])
            rec, prec, ap = voc_eval(
                pred_boxes, gt_boxes, ovthresh=iou_threshold
            )
            aps[class_name] = ap
        return aps


def parse_voc_annotations(annotations_dir):
    annotations = {}
    for annotation_file in os.listdir(annotations_dir):
        if not annotation_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(annotations_dir, annotation_file))
        objects = []
        for obj in tree.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            x_min = float(bbox.find("xmin").text)
            y_min = float(bbox.find("ymin").text)
            x_max = float(bbox.find("xmax").text)
            y_max = float(bbox.find("ymax").text)
            objects.append({"name": name, "bbox": [x_min, y_min, x_max, y_max]})
        image_id = os.path.splitext(annotation_file)[0]
        annotations[image_id] = objects
    return annotations


# Base Class supporting inference for a single test object
class ZeroShotObjectDetection:
    def __init__(self, clip_model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        self.faster_rcnn = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(self.device) # TODO: Add handling for multiple types of rcnns
        self.faster_rcnn.eval() # hv to run in inference mode

        os.makedirs("viz",exist_ok=True)
    
    def store_predictions(self, predictions, image_id, class_name, boxes, scores):
        for box, score in zip(boxes, scores):
            x_min, y_min, x_max, y_max = box
            # VOC requires +1 coordinate adjustments
            x_min, y_min = x_min + 1, y_min + 1
            predictions[class_name].append(
                f"{image_id} {score:.3f} {x_min:.1f} {y_min:.1f} {x_max:.1f} {y_max:.1f}"
            )

    def detect(self,image_path,text_queries,score_threshold=0.5):
        image = Image.open(image_path).convert("RGB")
        # image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)


        frcnn_result = self.compute_faster_rcnn_result(image)
        all_region_proposals = frcnn_result['all_region_proposal_boxes'].detach()
        candidates_data = frcnn_result['candidates']

        candidate_boxes = candidates_data['boxes'].detach()
        candidate_scores = candidates_data['scores'].detach()

        print(f"Total number of region proposal boxes: {all_region_proposals.shape[0]}")
        print(f"Total number of candidate boxes: {candidate_boxes.shape[0]}")

        self.plot_candidate_scores(candidate_scores, text_queries[0].split(" ")[1])
        self.plot_image_with_boxes(image,all_region_proposals)


        tensor_transform = transforms.ToTensor()
        image_pt = tensor_transform(image)
        height, width = image_pt.shape[-2:]

        max_similarity = -np.inf
        max_rel_box = None
        self.clip_model.eval()
        
        
        tokenized_query = clip.tokenize(text_queries).to(self.device)
        text_features = self.clip_model.encode_text(tokenized_query)
        norm_text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1).mean(dim=0, keepdim=True)

        for box in candidate_boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            x_max, y_max = min(width, x_max), min(height, y_max)

            if x_min > x_max or x_min > width or y_min > y_max or y_min > height:
                continue

            cropped_image = image_pt[:, y_min:y_max+1, x_min:x_max+1]
            if not torch.prod(torch.tensor(cropped_image.shape)):
                continue

            cropped_image = self.clip_preprocess(transforms.functional.to_pil_image(cropped_image)).unsqueeze(0).to(self.device)

            if not torch.prod(torch.tensor(cropped_image.shape)):
                continue
            
            image_features = self.clip_model.encode_image(cropped_image)
            norm_image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

            # norm_image_features = norm_image_features.unsqueeze(0)
            # norm_text_features = norm_text_features if norm_text_features.dim() == 2 else norm_text_features.unsqueeze(0)

            similarity = torch.dot(norm_image_features.view(-1), norm_text_features.view(-1))
            # print("Similarity", similarity)
            
            if similarity > 0.22:
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_rel_box = box
        
        return max_rel_box.cpu().numpy() if max_rel_box is not None else None


    def plot_image_with_boxes(self, image, boxes, labels=None, figsize=(8,6), save_name="default"):
        fig, ax = plt.subplots(1, figsize=(8,6))
        
        ax.imshow(image)
        
        for i, box in enumerate(boxes):

            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()

            x_min, y_min, x_max, y_max = box
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor="green",
                facecolor="none",
            )
            ax.add_patch(rect)

            if labels is not None:
                label_size = len(labels[i]) * 10
                ax.text(
                    x_min + (x_max - x_min) / 2 - label_size / 2,
                    y_min - 10,
                    labels[i],
                    fontsize=12,
                    verticalalignment="top",
                    color="white",
                    bbox=dict(facecolor="green", alpha=0.5, edgecolor="none"),
                )

        plt.tight_layout(pad=0)
        plt.axis('off')
        plt.savefig(f"viz/output_with_boxes_{save_name}")
        plt.close()
        

    def plot_candidate_scores(self,candidate_scores: torch.Tensor, save_label="default") -> None:
        plt.figure(figsize=(8, 6))
        plt.plot(candidate_scores.cpu().numpy(), marker="o", linestyle='-', color='b', markersize=5, linewidth=1.5)
        plt.xlabel("Candidate Box Index", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Candidate Box Scores", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"viz/candidate_scores_{save_label}.png")
        plt.close()

    def compute_faster_rcnn_result(self, image: Image.Image) -> dict:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        image_pt = transform(image).to(self.device)
        with torch.no_grad():
            transformed = self.faster_rcnn.transform([image_pt])[0]

        transformed.tensors = transformed.tensors.to(self.device)
        features = self.faster_rcnn.backbone(transformed.tensors)

        all_region_proposal_boxes = self.faster_rcnn.rpn(transformed, features)[0][0]
        frcnn_outputs = self.faster_rcnn(image_pt.unsqueeze(0))[0]

        return {
            "all_region_proposal_boxes": all_region_proposal_boxes,
            "candidates": frcnn_outputs,
        }

    

# inherits, extends functionality for larger datasets for speed and scalability
class ZeroShotObjectDetectionDatasets(ZeroShotObjectDetection):
    def __init__(self, clip_model_name="ViT-B/32", device=None):
        super().__init__(clip_model_name, device)

    def eval(self, image_paths, annotations_path, text_queries, score_threshold=0.5):
        predictions = defaultdict(list)  # Store predictions
        for image_path in tqdm(image_paths):
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            for class_name, prompts in text_queries.items():
                pred_box = self.detect(image_path, prompts, score_threshold=score_threshold)
                if pred_box is not None:
                    # VOC format
                    self.store_predictions(predictions, image_id, class_name, [pred_box], [1.0]) 

        return predictions


def parse_args():
    parser= argparse.ArgumentParser(description="Zero Shot Inference using minimal RPN approach")
    parser.add_argument("--image_path",type=str,required=False,help="path to image for eval")
    # parser.add_argument("--output_test_path",type=str,required=True,help="path to save test_image")
    # parser.add_argument("--text_queries",type=str,nargs='+',required=True,help="List of text queries for object detection (e.g., 'a cat', 'a jumping cat, etc').")
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Score threshold for filtering detections (default: 0.5).",
    )

    parser.add_argument("--debug",action="store_true",help="Test detector on a single image specified with --image_path, saved to './viz' dir")
    parser.add_argument("--dataset_name",required=False,help="Name of the dataset (e.g., VOC 2007, VOC 2012)")
    parser.add_argument("--image_dir", type=str, required=False, help="Directory containing the dataset images")
    parser.add_argument("--annotations_dir", type=str, required=False, help="Directory containing dataset annotations")

    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # detections = detector.detect(args.image_path, pascal_classes, score_threshold=args.score_threshold)
    # detector.plot_detections(args.image_path, detections, output_path=args.output_test_path)

    if (args.debug):
        detector = ZeroShotObjectDetection()
        matched_boxes=[]
        image = Image.open(args.image_path).convert("RGB")

        for label, prompts in label_prompts.items():
            print(f"Detecting {label}...")
            pred_box = detector.detect(args.image_path, prompts)
            if pred_box is None:
                print(f"No detection found at {label}")
                continue
                
            matched_boxes.append([pred_box,label])
            detector.plot_image_with_boxes(image, [pred_box], [label], save_name=f"{label}")


        detector.plot_image_with_boxes(image, [box[0] for box in matched_boxes], [box[1] for box in matched_boxes], save_name="final")
    elif args.dataset_name.lower() == 'pascal' or args.dataset_name.lower() == 'pascal_mini':
        image_dir = args.image_dir
        annotations_dir = args.annotations_dir
        image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".jpg")]
        annotations = parse_voc_annotations(annotations_dir)
        # print(annotations)
        detector = ZeroShotObjectDetectionDatasets()
        predictions = detector.eval(image_paths, annotations_dir, label_prompts)
        # evaluator = PascalVOCDetectionEvaluator(annotations, pascal_classes)
        # aps = evaluator.evaluate(predictions)
        # print(f"AP Results: {aps}")
    else:
        print("Not yet supported")

if __name__ == "__main__":
    main()