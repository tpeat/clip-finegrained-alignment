from clip_ods import CLIPDetectorV0
from PIL import Image
import json
from clip import load
from tqdm import tqdm
import torch
from pycocotools.coco import COCO
import os

print("CUDA AVAILABILITY", torch.cuda.is_available())

coco_dir = "/storage/ice1/9/3/kkundurthy3/dataset/coco"
val_ann_file = os.path.join(coco_dir, "annotations/instances_val2017.json")
val_coco = COCO(val_ann_file)

categories = {cat['id']: cat['name'] for cat in val_coco.loadCats(val_coco.getCatIds())}
print(categories)



with open("/storage/ice1/9/3/kkundurthy3/synthetic_dataset/synthetic_annotations.json", "r") as f:
    synthetic_data = json.load(f)


# print(synthetic_data)

# for data_point in synthetic_data:
#     image_path = data_point["image_path"]
#     width,height = data_point["width"], data_point["height"]
#     caption = data_point["caption"]
#     source_object = data_point["source_object"]

model, preprocess = load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = CLIPDetectorV0(model=model, transforms=preprocess, device=device)


detection_results = []


ctr = 0
for data_point in tqdm(synthetic_data, desc="Processing images"):
    try:
        image_path = data_point["image_path"]
        width,height = data_point["width"], data_point["height"]
        caption = data_point["caption"]
        # source_object = data_point["source_object"]
        
        img = Image.open(image_path).convert("RGB")
        # img = preprocess(img).unsqueeze(0).to(device)

        img_width, img_height = img.size
        grid_size=128
        anchor_coords = [
            [x, y, x + grid_size, y + grid_size] for x in range(0, img_width, grid_size) for y in range(0, img_height, grid_size)
        ]
        anchor_features = detector.get_anchor_features(img, anchor_coords, bs=128)
        print("ANCHOR FEATURES", anchor_features)

        detection_results_by_category = {}
        for cat_id,category in categories.items():
            detection_img, detection_result, thr = detector.detect_by_text(
                texts=[category],
                img=img,
                coords=anchor_coords,
                anchor_features=anchor_features,
                tp_thr=0.5,  # tp threshold
                fp_thr=-2.0,  # fp
                iou_thr=0.3,  # IOU
                skip_box_thr=0.5,  # Confidenc
            )

            detection_results_by_category[category] = {
                "detections": detection_result,
                "threshold": thr,
            }

            print(detection_result)

            print(categories.keys())
            for box, score, label in zip(
                detection_result["boxes"], detection_result["scores"], detection_result["labels"]
            ):
                category_name = categories[int(label)]
                print(f"Detected {category_name} with score {score:.2f} at box {box}")


        detection_results.append({
            "image_path": image_path,
            "ground truth": caption,
            "resolution": f"{width}x{height}",
            "detections_by_category": detection_results_by_category,
        })

        if ctr % 5 == 0:
            result_path = f"/storage/ice1/9/3/kkundurthy3/synthetic_dataset/detection_eval_result_{ctr}.json"
            with open(result_path, "w") as f:
                json.dump(detection_results, f, indent=4)


        
        print(f"Processed {image_path}")
        print("________________________________________________________________________________")

        ctr += 1

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        raise e
        continue
    

results_path = "/storage/ice1/9/3/kkundurthy3/synthetic_dataset/detection_eval_results.json"
with open(results_path, "w") as f:
    json.dump(detection_results, f, indent=4)



print(f"Detection results saved to {results_path}")


