from dataclasses import dataclass
from typing import Tuple

@dataclass
class CLIPFineTuneConfig:
    """Configuration for CLIP finetuning"""
    lr: float = 5e-5
    batch_size: int = 32
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    use_amp: bool = True  # Mixed precision training
    clip_model: str = "openai/clip-vit-base-patch32"
    max_length: int = 77  # CLIP's max token length
    experiment_name: str = "clip_default"
    gradient_accumulation_steps: int = 1
    loss_type: str = "clip"  # "clip" or "sparc"
    similarity_threshold: float = 0.5
    global_loss_weight: float = 1.0
    local_loss_weight: float = 1.0
    inverse_temperature: float = 0.07
    optimizer_type: str = "adamw"
    betas: Tuple[float, float] = (0.9, 0.999)  # Changed: Added type annotation
    eps: float = 1e-8
    amsgrad: bool = False
    