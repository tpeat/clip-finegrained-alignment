import os
import json
import argparse
import random
from shutil import copyfile
from PIL import Image, ImageDraw

upload_dir = "./test_images"

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
            draw.text((x1, y1), str(label), fill="yellow")

        visualized_path = os.path.join(args.output_dir, f"sample_{idx}_visualized.png")
        img.save(visualized_path)



if __name__ == "__main__":
    main()