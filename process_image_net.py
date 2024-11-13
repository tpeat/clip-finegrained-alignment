import kagglehub

# Download latest version
path = kagglehub.dataset_download("samrat230599/fastai-imagenet")

print("Path to dataset files:", path)