# @author: GPT 4
import os
import argparse
import urllib.request
import tarfile

# URL for Pascal VOC 2012 dataset
VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

def download_and_extract_voc(output_dir: str):
    """Download and extract the Pascal VOC dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    tar_path = os.path.join(output_dir, "VOCtrainval_11-May-2012.tar")
    
    # Download the dataset
    if not os.path.exists(tar_path):
        print(f"Downloading Pascal VOC dataset to {tar_path}...")
        urllib.request.urlretrieve(VOC_URL, tar_path)
        print("Download complete.")
    else:
        print(f"Dataset already downloaded at {tar_path}.")
    
    # Extract the dataset
    extract_dir = os.path.join(output_dir, "VOCdevkit")
    if not os.path.exists(extract_dir):
        print(f"Extracting dataset to {output_dir}...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(output_dir)
        print("Extraction complete.")
    else:
        print(f"Dataset already extracted at {extract_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Download Pascal VOC dataset.")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to download and extract the Pascal VOC dataset."
    )
    args = parser.parse_args()
    
    download_and_extract_voc(args.output_dir)

if __name__ == "__main__":
    main()
