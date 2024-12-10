# from transformers import CLIPProcessor, CLIPModel
import torch
import clip

class_names = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

custom_weights_path = "/storage/ice1/9/3/kkundurthy3/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# inputs = processor(text=class_names, return_tensors="pt", padding=True)
# inputs = {key: value.to(device) for key, value in inputs.items()}


model, preprocess = clip.load("RN50", device=device)

# state_dict = torch.load(custom_weights_path, map_location=device)
# model.load_state_dict(state_dict)

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)

# with torch.no_grad():
#     embeddings = model.get_text_features(**inputs)

with torch.no_grad():
    text_features = model.encode_text(text_inputs)

# embeddings = embeddings.cpu()
# torch.save(embeddings, "/storage/ice1/9/3/kkundurthy3/RegionCLIP/pretrained_ckpt/concept_emb/pascal_cls_emb_vit.pth")
torch.save(text_features.cpu(), "/storage/ice1/9/3/kkundurthy3/RegionCLIP/pretrained_ckpt/concept_emb/pascal_cls_emb_rn.pth")