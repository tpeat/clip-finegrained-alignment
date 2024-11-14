import json
import os
import sys
from pycocotools.coco import COCO
import requests
import logging
import argparse
from PIL import Image
import random



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

validation_dir = "dataset/3/FastAI_ImageNet_v2/val"
img_net_classes = ["cassette_player", "chain_saw", "french_horn", "golf_ball", "parachute", "tench"]
img_save_dir = "dataset/coco_imgs"
count_to_small_path = "dataset/CountToSmallBench"


def parse_args():
    parser = argparse.ArgumentParser(description="Add arguments")
    parser.add_argument('--function', type=str, required=True, help="Build coco or build dataset") 
    return parser.parse_args()

def main(args):
    
    if (args.function=="coco"):
        coco = COCO("dataset/coco/annotations/instances_train2017.json")
        os.makedirs(img_save_dir, exist_ok=True)
        images = coco.loadImgs(coco.getImgIds())
        loaded = set([f for f in os.listdir(img_save_dir)])
        for img in images:
            name = img["file_name"]
            if img['file_name'] in loaded:
                logger.info(f"DOWNLOADED: {name} ALREADY")
            else:
                with open(os.path.join(img_save_dir, img["file_name"]),'wb') as file:
                    data= requests.get(img['coco_url']).content
                    file.write(data)
                
                logger.info(f"DOWNLOADED: {name}")
    if (args.function=="coco_val"):
        coco = COCO("dataset/coco/annotations/instances_val2017.json")
        pass
    elif (args.function=="data"):
        for file in os.listdir(img_save_dir):
            class_sample = random.sample(img_net_classes, 1)
            small_obj_class = class_sample[0]
            candidate_imgs= [f for f in os.listdir(f"{validation_dir}/{small_obj_class}")]
            count = random.randint(1,10)
            selected_imgs = random.sample(candidate_imgs,count)

            orig_img = Image.open(os.path.join(img_save_dir,file)).convert("RGBA")

            for img in selected_imgs:
                small_img = Image.open(os.path.join(validation_dir,small_obj_class,img)).convert("RGBA")
                dim = random.randint(16,32)

                small_img = small_img.resize((dim,dim))

                #placing img on orig img at random pos
                # would this lead to any interference if blocking important things in the image
                rand_x_pos = random.randint(0,orig_img.width - small_img.width)
                rand_y_pos = random.randint(0,orig_img.height - small_img.height)
                orig_img.paste(small_img,(rand_x_pos,rand_y_pos))


            save_path = os.path.join("dataset/final_images",f"{small_obj_class}_{count}_{file}")
            orig_img.convert("RGB").save(save_path)
            
            bench_ele = {
                "Question": f"How many {small_obj_class}(s) are in this image?", 
                "Class Label": f"{small_obj_class}", 
                "Count": f"{count}",
                "Image Path": f"{save_path}"
            }

            # at inference timem we are just iterating through this and getting the corresponding img path
            ele_path = os.path.join(count_to_small_path,f"{small_obj_class}_{count}_{file}".removesuffix(".jpg"))
            ele_path = f"{ele_path}.json"
            try:
                with open(ele_path,"x") as file:
                    json.dump(bench_ele, file, indent=4)
            except Exception:
                logger.info("PATH EXISTS, ADDING V2")
                ele_path = ele_path.removesuffix(".json") + "-2" + ".json"
                try:
                    with open(ele_path,"x") as file:
                        json.dump(bench_ele, file, indent=4)
                except Exception as e:
                    logger.info(f"{e}")
                    pass

            
            pass
        
    elif (args.function=="data_val"):
        pass
    else:
        pass

if __name__ == "__main__":
    args=parse_args()
    main(args)


    
