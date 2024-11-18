# @author: Claude

import os
import requests
from tqdm import tqdm
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COCODownloader:
    def __init__(self, base_dir="dataset"):
        self.base_dir = base_dir
        self.coco_dir = os.path.join(base_dir, "coco")
        self.annotation_dir = os.path.join(self.coco_dir, "annotations")
        
        os.makedirs(self.coco_dir, exist_ok=True)
        os.makedirs(self.annotation_dir, exist_ok=True)
        
        self.urls = {
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "train_images": "http://images.cocodataset.org/zips/train2017.zip",
            "val_images": "http://images.cocodataset.org/zips/val2017.zip",
        }

    def download_file(self, url, destination):
        """Download a file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        logger.info(f"Downloading {url} to {destination}")
        
        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

    def extract_zip(self, zip_path, extract_path):
        """Extract a zip file"""
        logger.info(f"Extracting {zip_path} to {extract_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        os.remove(zip_path)
        logger.info(f"Removed {zip_path}")

    def download_annotations(self):
        """Download and extract COCO annotations"""
        zip_path = os.path.join(self.coco_dir, "annotations.zip")
        
        self.download_file(self.urls["annotations"], zip_path)
        
        self.extract_zip(zip_path, self.coco_dir)
        
        logger.info("Annotations downloaded and extracted successfully")

    def download_images(self, split="train"):
        """Download and extract COCO images for a specific split"""
        url_key = f"{split}_images"
        if url_key not in self.urls:
            raise ValueError(f"Invalid split: {split}. Choose 'train' or 'val'")
        
        zip_path = os.path.join(self.coco_dir, f"{split}2017.zip")
        
        self.download_file(self.urls[url_key], zip_path)
        
        self.extract_zip(zip_path, self.coco_dir)
        
        logger.info(f"{split.capitalize()} images downloaded and extracted successfully")


def main():
    downloader = COCODownloader()
    
    downloader.download_annotations()
    
    downloader.download_images("train")


if __name__ == "__main__":
    main()