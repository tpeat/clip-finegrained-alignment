import json
import os
import argparse
import torch
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import hog
from skimage import color
import cv2
import json
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import selectivesearch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_labels = {
    0: 'no object detected',
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

class ZeroShotObjectDetector:
    def __init__(self,model_path="pretrained",rpn_method="selective_search",max_iter=50):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_path == "pretrained":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_path)
        else:
            # will update this to load the model weights from the provided path
            self.model = None
            self.clip_processor = None

        self.max_iter = max_iter
        
        self.rpn_method = rpn_method

    # just testing on the images we loaded into the local dir
    def test_selective_search(self, input_dir,output_dir):
        os.makedirs(output_dir, exist_ok=True)
        img_files= [f for f  in os.listdir(input_dir) if (f.endswith(".png") and "visualized" not in f)]
        print("USING FILES: ", img_files)

        for idx, file in enumerate(img_files):
            input_img_pth = os.path.join(input_dir, file)
            output_img_pth = os.path.join(output_dir, f"region_proposal_selective_search_{idx}.png")

            img = cv2.cvtColor(cv2.imread(input_img_pth), cv2.COLOR_BGR2RGB)

            # actually getting the regions
            img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
            
            # filtering
            candidates=set()
            for r in regions:
                if r['rect'] in candidates:
                    continue
                if r['size'] < 1500: # random number, just to make sure we don't consider everything
                    continue
                
                x,y,w,h=r['rect']
                # based on color variation
                if w == 0 or h == 0:
                    continue
                candidates.add(r['rect'])

            pil_image = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_image)
            for x, y, w, h in candidates:
                draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
            
            pil_image.save(output_img_pth)


    """
    STAGE 1 of the pipeline
    """
    def extract_clip_features(self,image_path):
        image = Image.open(image_path).convert("RGBA")
        inputs = self.clip_processor(images=image,return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_feats = self.model()
        
        return img_feats.cpu().numpy()

    """
    STAGE 2 of the pipeline, using rpn method
    """
    def generate_proposals(self,image_path):
        if self.rpn_method == "selective_search":
            return self.selective_search(image_path)
        else:
            raise Exception("pls specify selective search or hog")

    def selective_search(self,image_path):
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        return ss.process() # this crresponds to rectanlges
    


    


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot object detection with CLIP.")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                        help="default: openai/clip-vit-base-patch32, else provide path to finetuned CLIP weights.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset directory containing images.")
    parser.add_argument("--rpn", type=str, default="selective_search",
                        choices=["selective_search", "hog"],
                        help="Method for regional proposal generation (default: selective_search).")
    parser.add_argument("--output_dir", type=str, default="./test_images",
                        help="Local directory to save results and intermediate outputs (default: ./test_images).")
    parser.add_argument("--test_pipeline", type=bool, default=True,
                        help="Apply pipeline only to test images")
    return parser.parse_args()

def main(args):
    model = args.model
    dataset_path = args.dataset_path
    rpn = args.rpn
    output_dir = args.output_dir
    detector = ZeroShotObjectDetector(model_path=model,rpn_method=rpn)
    if args.test_pipeline:
        detector.test_selective_search(output_dir,output_dir)

def extract_clip_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features.cpu().numpy()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using model: {args.model}")
    main(args)