import os
import json
import argparse
import random
from shutil import copyfile
from PIL import Image, ImageDraw

upload_dir = "./test_images"

clip_labels = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird',
    17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
    36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Sample synthetic images and annotations.")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to the JSON file containing validation data.")
    parser.add_argument("--output_dir", type=str, required=False, default="./test_images",
                        help="Directory to save the sampled images and annotations.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of images to sample (default: 5).")
    return parser.parse_args()

def main():
    args = parse_args()
    upload_dir = args.output_dir
    os.makedirs(upload_dir,exist_ok=True)
    
    with open(args.json_path, "r") as f:
        validation_data = json.load(f)

    annotations = validation_data['annotations']
    sampled_images = random.sample(annotations, args.num_samples)

    for idx, sample in enumerate(sampled_images):
        # Copy img to the output directory
        src_image_path = sample['image_path']
        dst_image_path = os.path.join(upload_dir, f"sample_{idx}.png")
        copyfile(src_image_path, dst_image_path)

        # Save annotations as new JSON file
        annotation_path = os.path.join(args.output_dir, f"sample_{idx}_annotation.json")
        with open(annotation_path, "w") as ann_file:
            json.dump(sample, ann_file, indent=4)

        img = Image.open(src_image_path)
        draw = ImageDraw.Draw(img)
        
        for box, label in zip(sample['boxes'], sample['labels']):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            label_class = clip_labels[label]
            draw.text((x1, y1), str(label_class), fill="yellow")

        visualized_path = os.path.join(args.output_dir, f"sample_{idx}_visualized.png")
        img.save(visualized_path)



if __name__ == "__main__":
    main()