import clip
import torch
from collections import OrderedDict

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Extract the state dict for the CLIP model
state_dict = model.state_dict()

# Transform state_dict keys for compatibility with RegionCLIP
new_state_dict = OrderedDict()
for key, value in state_dict.items():
    new_key = f"visual_encoder.{key}"  # Prefix for visual encoder
    new_state_dict[new_key] = value

# Add RPN weights
rpn_weights_path = "/storage/ice1/9/3/kkundurthy3/RegionCLIP/pretrained_ckpt/rpn/rpn_lvis_1203_clipbackbone.pth"
rpn_weights = torch.load(rpn_weights_path, map_location=device)

# Combine RPN weights into the state dict
for key, value in rpn_weights.items():
    new_state_dict[f"region_proposal_network.{key}"] = value

# Initialize task-specific head weights
# Classification head for region proposals
classification_head = {
    "cls_score.weight": torch.randn(81, 1024),  # Example: 81 classes, 1024-dim features
    "cls_score.bias": torch.zeros(81),
    "bbox_pred.weight": torch.randn(320, 1024),  # Example: bbox regression params
    "bbox_pred.bias": torch.zeros(320),
}

# Add classification head to the state dict
for key, value in classification_head.items():
    new_state_dict[f"roi_heads.box_predictor.{key}"] = value

# Mask prediction head (if needed, for instance segmentation tasks)
mask_head = {
    "deconv.weight": torch.randn(256, 256, 4, 4),  # Example: Deconvolution layer
    "predictor.weight": torch.randn(81, 256, 1, 1),  # 81 classes, 1x1 conv for masks
    "predictor.bias": torch.zeros(81),
}

# Add mask head to the state dict (if required)
for key, value in mask_head.items():
    new_state_dict[f"roi_heads.mask_head.{key}"] = value

# Trainer metadata
trainer_data = {
    "iteration": 0,
    "hooks": {},
}

# Construct the full checkpoint
checkpoint = {
    "model": new_state_dict,
    "trainer": trainer_data,
    "iteration": 0,
    "metadata": {
        "model_name": "RegionCLIP-ViT-B/32",
        "pretrained_dataset": "LVIS",
        "architecture": "RegionCLIP",
        "input_resolution": 224,
    },
}

# Save the checkpoint to disk
output_path = "/storage/ice1/9/3/kkundurthy3/RegionCLIP/pretrained_ckpt/regionclip/clip_vit_b32_with_rpn_heads.pth"
torch.save(checkpoint, output_path)

print(f"Checkpoint saved to {output_path}")
