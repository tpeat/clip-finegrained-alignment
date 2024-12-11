from dataclasses import dataclass
from typing import Tuple

@dataclass
class CLIPFineTuneConfig:
    """Configuration for CLIP finetuning"""
    lr: float = 1e-5
    batch_size: int = 32
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    max_epochs: int = 400
    save_every: int = 1
    weight_decay: float = 0.2
    use_amp: bool = True # mixed precision
    clip_model: str = "ViT-B/32"
    max_length: int = 77  # CLIP's max token length
    experiment_name: str = "clip_default"
    gradient_accumulation_steps: int = 4
    loss_type: str = "count"  # "clip" or "sparc"
    similarity_threshold: float = 0.5
    global_loss_weight: float = 1.0
    local_loss_weight: float = 1.0
    inverse_temperature: float = 1.0
    optimizer_type: str = "adamw"
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 5e-6
    amsgrad: bool = False
    count_alpha: float = 1.0

    def print_config(self):
        """Print training configuration in an organized format"""
        print("\n" + "="*50)
        print("TRAINING CONFIGURATION")
        print("="*50)

        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        training_params = {
            "Training Hyperparameters": {
                "Learning Rate": self.lr,
                "Batch Size": self.batch_size,
                "Gradient Accumulation Steps": self.gradient_accumulation_steps,
                "Effective Batch Size": effective_batch_size,
                "Max Gradient Norm": self.max_grad_norm,
                "Warmup Steps": self.warmup_steps,
                "Weight Decay": self.weight_decay,
                "Mixed Precision": self.use_amp
            },
            "Model Configuration": {
                "CLIP Model": self.clip_model,
                "Max Token Length": self.max_length,
                "Experiment Name": self.experiment_name,
                "Loss Type": self.loss_type
            },
            "Loss Parameters": {
                "Count Alpha": self.count_alpha if self.loss_type == "count" else "N/A",
                "Similarity Threshold": self.similarity_threshold if self.loss_type == "sparc" else "N/A",
                "Global Loss Weight": self.global_loss_weight if self.loss_type == "sparc" else "N/A",
                "Local Loss Weight": self.local_loss_weight if self.loss_type == "sparc" else "N/A",
                "Inverse Temperature": self.inverse_temperature
            },
            "Optimizer Configuration": {
                "Type": self.optimizer_type,
                "Betas": self.betas,
                "Epsilon": self.eps,
                "AMSGrad": self.amsgrad
            }
        }

        for group_name, params in training_params.items():
            print(f"\n{group_name}:")
            for param_name, value in params.items():
                print(f"  {param_name}: {value}")

        print("\n" + "="*50 + "\n")
    