"""
Performs the following operations on PASCAL

- Visualize the distribution of object sizes and classes in pascal
- Create PascalSubset PASCAL with all images containing bounding boxes 
- Create PascalMini: Resizing larger bounding boxes

Small <= 64 x 64
Medium <= 96x96
Large > 96 x 96

"""

import os
import xml.etree.ElementTree as ET
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageFilter


def parse_args():
    parser= argparse.ArgumentParser(description="visualize pascal ")
    parser.add_argument("--create_mini",action="store_true", default=False,help="Create Pascal Mini benchmark at same directory as pascal installation")
    parser.add_argument(
        "--pascal_path", type=str, help="path to pascal installation"
    )
    
    return parser.parse_args()

box_counts={"small":0, "medium":0, "large":0}
small_img_files = {}
classes = defaultdict(int)
total = 0


def resize_image(boxes_to_resize, mini_path, anno_path):
    if len(boxes_to_resize) == 0:
        return
    
    img_path = boxes_to_resize[0]["img_path"]
    file_name = os.path.basename(img_path)
    parent_dir = os.path.dirname(img_path)
    print("IMG: ",file_name)
    img = Image.open(img_path)
    img_copy = img.copy()

    tree = ET.parse(anno_path)
    root = tree.getroot()

    boxes_set = [box["box"] for box in boxes_to_resize]

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        xmax_resize = (xmax-xmin)//2
        ymax_resize = (ymax-ymin)//2

        if [xmin,ymin,xmax,ymax] in boxes_set:
            print("Resizing")
            bbox_area = img.crop((xmin, ymin, xmax, ymax))
            bbox_resized = bbox_area.resize(
                (bbox_area.width // 2, bbox_area.height // 2), Image.Resampling.LANCZOS
            )
            blurred_area = img_copy.crop((xmin, ymin, xmax, ymax)).filter(ImageFilter.GaussianBlur(10))
            img.paste(blurred_area, (int(xmin), int(ymin), int(xmax), int(ymax)))


            new_xmin = xmin + (bbox_area.width - bbox_resized.width) // 2
            new_ymin = ymin + (bbox_area.height - bbox_resized.height) // 2
            new_xmax = new_xmin + bbox_resized.width
            new_ymax = new_ymin + bbox_resized.height

            img.paste(bbox_resized, (int(new_xmin), int(new_ymin)))
    
    # test_path = f"visualizations/{file_name}"

    # if random.random() < 0.2:
    #     img.save(test_path)

    path = os.path.join(parent_dir,f"resize-{file_name}")

    img.save(path)
    return path

            



def parseImage(mini_path, anno_path):
    root = ET.parse(anno_path).getroot()
    img_name = root.find("filename").text
    img_path = os.path.join(mini_path,f"JPEGImages/{img_name}")


    boxes_to_resize = []
    for obj in root.findall("object"):
        bndbox=obj.find("bndbox")

        xmin=float(bndbox.find("xmin").text)
        ymin=float(bndbox.find("ymin").text)
        xmax=float(bndbox.find("xmax").text)
        ymax=float(bndbox.find("ymax").text)
        width, height = xmax - xmin, ymax - ymin
        

        if width*height < 32 * 32:
            pass
        elif 32 * 32 <= width*height < 96 * 96:
            if random.random() > 0.5:
                new_width = width // 2
                new_height = height // 2
                new_xmin = xmin + (width - new_width) / 2
                new_ymin = ymin + (height - new_height) / 2
                new_xmax = new_xmin + new_width
                new_ymax = new_ymin + new_height

                bndbox.find("xmin").text = str(new_xmin)
                bndbox.find("ymin").text = str(new_ymin)
                bndbox.find("xmax").text = str(new_xmax)
                bndbox.find("ymax").text = str(new_ymax)

                box_obj = {"img_path": img_path, "box": [xmin,ymin,xmax,ymax],"class": obj.find("name").text}
                boxes_to_resize.append(box_obj)
                
            else:
                pass
        else:
            if random.random() > 0.8:
                new_width = width // 2
                new_height = height // 2
                new_xmin = xmin + (width - new_width) / 2
                new_ymin = ymin + (height - new_height) / 2
                new_xmax = new_xmin + new_width
                new_ymax = new_ymin + new_height

                bndbox.find("xmin").text = str(new_xmin)
                bndbox.find("ymin").text = str(new_ymin)
                bndbox.find("xmax").text = str(new_xmax)
                bndbox.find("ymax").text = str(new_ymax)
                box_obj = {"img_path": img_path, "box": [xmin,ymin,xmax,ymax],"class": obj.find("name").text}
                boxes_to_resize.append(box_obj)
                
            else:
                pass

    print("TO RESIZE: ", boxes_to_resize)

    if (len(boxes_to_resize) > 0):
        path = resize_image(boxes_to_resize, mini_path, anno_path)
        root.find("filename").text = path
        tree = ET.ElementTree(root)
        os.makedirs(os.path.join(mini_path, "Annotations"),exist_ok=True)
        updated_anno_path = os.path.join(mini_path, f"Annotations/new_{os.path.basename(anno_path)}")
        tree.write(updated_anno_path)
    

    

    



# parsin for single path
def parse_annotations(anno_path):
    root = ET.parse(anno_path).getroot()

    # for child in root:
    #     print(child.tag, child.attrib)
    total_obj = 0
    for obj in root.findall("object"):
        bndbox=obj.find("bndbox")
        xmin=float(bndbox.find("xmin").text)
        ymin=float(bndbox.find("ymin").text)
        xmax=float(bndbox.find("xmax").text)
        ymax=float(bndbox.find("ymax").text)

        width, height = xmax - xmin, ymax - ymin
        
        # 48 is a less strict threshold to record small images
        if width * height < 48 * 48:
            if anno_path in small_img_files.keys():
                small_img_files[anno_path] += 1
            else:
                small_img_files[anno_path] = 1
            
            

        if width*height < 32 * 32:
            size_cat = "small"
        elif 32 * 32 <= width*height < 96 * 96:
            size_cat = "medium"
        else:
            size_cat = "large"

        
        box_counts[size_cat] += 1
        classes[obj.find("name").text] += 1
        total_obj += 1


    return total_obj
  
def viz(size=True):
    if size:
        labels = box_counts.keys()
        sizes = [box_counts[label] for label in labels]
    else:
        labels = classes.keys()
        sizes = [classes[label] for label in labels]
    fig,ax = plt.subplots(figsize=(20,12))
    ax.pie(sizes,labels=labels,autopct='%1.1f%%')
    ax.axis('equal')
    name = "Size" if size else "Classes"
    plt.title(f"{name} Distribution")
    plt.savefig(f"visualizations/{name} Distribution")
     

def main():
    args=parse_args()
    if(args.create_mini):
        print("Creating Pascal Mini ...")
        parent_dir = os.path.dirname(args.pascal_path)
        mini_dir = os.path.join(parent_dir,"pascalMini/VOCdevkit/VOC2012/")
        anno_dir = os.path.join(parent_dir,"pascalMini/VOCdevkit/VOC2012/OG_Annotations")
        print(mini_dir)
        print(anno_dir)
        for file in os.listdir(anno_dir):
            if file.startswith("new") or file.startswith("updated"):
                continue
            
            print(file)
            anno_path = os.path.join(anno_dir,file)
            parseImage(mini_dir,anno_path)

    else:
        anno_dir = os.path.join(args.pascal_path,"VOCdevkit/VOC2012/Annotations")
        total= 0
        for file in os.listdir(anno_dir):
            anno_path = os.path.join(anno_dir,file)
            total += parse_annotations(anno_path)
        print(f"TOTAL BOXES: {total}")
        print(box_counts)
        print(classes)

        with open("visualizations/small_obj_files.txt","a") as f:
            for img in small_img_files.keys():
                f.write(img + "\n")
        

        viz()
        viz(size=False)






if __name__ == "__main__":
    main()