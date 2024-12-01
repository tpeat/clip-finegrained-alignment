import argparse
import json
import os
from PIL import Image
from tqdm import tqdm
import torch
from pycocotools.coco import COCO
from clip_ods import CLIPDetectorV0, CLIPDetectorV1
from clip import load


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-based object detection")
    parser.add_argument("--coco_dir", type=str, required=True, help="Path to COCO dataset directory")
    parser.add_argument("--synthetic_ann_file", type=str, required=True, help="Path to synthetic annotations JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_version", type=str, choices=["v0", "v1"], default="v0",
                        help="Model version to use (v0 or v1)")
    parser.add_argument("--pretrained_model", type=str, default="ViT-B/32",
                        help="Specify the CLIP pretrained model to use (e.g., ViT-B/32, RN50)")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Threshold for bounding box confidence")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold for NMS")
    parser.add_argument("--fine_tuned_model", type=str, default=None,
                        help="Path to fine-tuned CLIP model weights (optional)")
    return parser.parse_args()


def initialize_detector(pretrained_model, version, device,fine_tuned_model=None):
    print(f"Loading CLIP model: {pretrained_model} of version {version}")
    model, preprocess = load(pretrained_model, device=device, jit=False)
    if fine_tuned_model:
        print(f"Attempting to load fine-tuned model weights from {fine_tuned_model}")
        try:
            state_dict = torch.load(fine_tuned_model, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Failed to load fine-tuned model. Error: {e}")
            raise e
    
    if version == "v0":
        return CLIPDetectorV0(model=model, transforms=preprocess, device=device), preprocess
    elif version == "v1":
        return CLIPDetectorV1(model=model, transforms=preprocess, device=device), preprocess
    else:
        raise ValueError("Invalid model version specified.")

# some gpt support
def process_image(data_point, detector, categories, args):
    """Process a single image and perform detection."""

    if not isinstance(data_point, dict):
        raise TypeError(f"Expected data_point to be a dictionary, got {type(data_point)}")

    image_path = data_point["image_path"]
    image_path = data_point["image_path"]
    width, height = data_point["width"], data_point["height"]
    caption = data_point["caption"]

    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size

    grid_size = 128
    anchor_coords = [
        [x, y, x + grid_size, y + grid_size]
        for x in range(0, img_width, grid_size)
        for y in range(0, img_height, grid_size)
    ]

    # Extract anchor features
    anchor_features = detector.get_anchor_features(img, anchor_coords, bs=128)

    detection_results_by_category = {}
    for cat_id, category in categories.items():
        detection_img, detection_result, thr = detector.detect_by_text(
            texts=[category],
            img=img,
            coords=anchor_coords,
            anchor_features=anchor_features,
            tp_thr=args.confidence_threshold,  # True Positive threshold
            fp_thr=-2.0,  # False Positive threshold
            iou_thr=args.iou_threshold,  # IOU threshold
            skip_box_thr=args.confidence_threshold,  # Confidence threshold
        )

        # Handle low-confidence detections, and project to class 0
        filtered_results = {
            "boxes": [],
            "scores": [],
            "labels": [],
        }
        for box, score, label in zip(
            detection_result["boxes"], detection_result["scores"], detection_result["labels"]
        ):
            if score < args.confidence_threshold:
                # Assign category 0 (no region) for low-confidence predictions
                filtered_results["boxes"].append([0, 0, 0, 0])
                filtered_results["scores"].append(0.0)
                filtered_results["labels"].append(0)
            else:
                filtered_results["boxes"].append(box)
                filtered_results["scores"].append(score)
                filtered_results["labels"].append(label)

        detection_results_by_category[category] = {
            "detections": filtered_results,
            "threshold": thr,
        }

    return {
        "image_path": image_path,
        "ground truth": caption,
        "resolution": f"{width}x{height}",
        "detections_by_category": detection_results_by_category,
    }


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA Available: {torch.cuda.is_available()}")
    file_path = "/home/hice1/kkundurthy3/CountDetectVLM/clip-finegrained-alignment/checkpoints/epoch_41.pt"

    try:
        state_dict = torch.load(file_path, map_location="cpu")
        if isinstance(state_dict, dict):
            print("The file is a state dictionary.")
        else:
            print("The file is not a state dictionary.")
    except Exception as e:
        print(f"Failed to load the file. Error: {e}")

    detector, preprocess = initialize_detector(
        args.pretrained_model,
        args.model_version,
        device,
        fine_tuned_model=args.fine_tuned_model
    )

    val_ann_file = os.path.join(args.coco_dir, "annotations/instances_val2017.json")
    val_coco = COCO(val_ann_file)
    categories = {cat['id']: cat['name'] for cat in val_coco.loadCats(val_coco.getCatIds())}

    with open(args.synthetic_ann_file, "r") as f:
        synthetic_data = json.load(f)

    

    os.makedirs(args.output_dir, exist_ok=True)
    detection_results = {
        "meta": {
            "pretrained_model": args.pretrained_model,
            "model_version": args.model_version,
            "confidence_threshold": args.confidence_threshold,
            "iou_threshold": args.iou_threshold,
        },
        "results": []
    }
    for ctr, data_point in enumerate(tqdm(synthetic_data['annotations'], desc="Processing images")):
        try:
            result = process_image(data_point, detector, categories, args)
            detection_results["results"].append(result)

            model_name = args.pretrained_model.replace("/", "_")
            if ctr % 5 == 0:
                result_path = os.path.join(args.output_dir, f"detection_eval_result_{ctr}_{model_name}_{args.model_version}.json")
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                with open(result_path, "w") as f:
                    json.dump(detection_results, f, indent=4)

        except Exception as e:
            path = data_point['image_path']
            print(f"Error processing {path}: {e}")
            continue

    results_path = os.path.join(args.output_dir, f"detection_eval_results_{args.pretrained_model}_{args.model_version}.json")
    with open(results_path, "w") as f:
        json.dump(detection_results, f, indent=4)

if __name__ == "__main__":
    main()