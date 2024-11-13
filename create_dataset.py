import json
import os
import sys
from pycocotools.coco import COCO
import requests
import logging
import argparse



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
1. Set output directory
2. Iterate through each coco image in test and validation
3. Randomly sample an image net class and randomly sample an object within that class
4. Randomly Resize the image to something between 16 x 16 and 32 x 32
5. Randomly overlap the resized ImageNet object over the coco object
6. Save the image to the output directory and create the CountBench Json format, with image attribute set to where the image was saved
'''

coco = COCO("dataset/coco/annotations/instances_train2017.json")
validation_dir = "dataset/3/FastAI_ImageNet_v2/val"
img_net_classes = ["cassette_player", "chain_saw", "church", "english_springer", "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute", "tench"]
img_save_dir = "dataset/coco/imgs"


def parse_args():
    parser = argparse.ArgumentParser(description="Add arguments")
    parser.add_argument('--function', type=str, required=True, help="Build coco or build dataset") 
    return parser.parse_args()

def main(args):
    
    if (args.function=="coco"):
        os.makedirs(img_save_dir, exist_ok=True)
        images = coco.loadImgs(coco.getImgIds())

        for img in images:
            with open(os.path.join(img_save_dir, img["file_name"]),'wb') as file:
                data= requests.get(img['coco_url']).content
                file.write(data)
            name = img["file_name"]
            logger.info(f"DOWNLOADED: {name}")
    if (args.function=="coco_val"):
        pass
    elif (args.function=="data"):
        for file in os.listdir(img_save_dir):
            pass
    elif (args.function=="data_val"):
        pass
    else:
        pass

if __name__ == "__main__":
    args=parse_args()
    main(args)


    
