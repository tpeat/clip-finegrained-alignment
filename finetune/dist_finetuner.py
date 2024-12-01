import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from transformers import CLIPModel
from tqdm import tqdm
import random
import numpy as np
import os
import time
import copy
import argparse

from dummy_data import create_coco_dataloaders
from losses import CustomCLIPLoss, SPARCLoss, CLIPCountLoss
from config import CLIPFineTuneConfig
from optimizers import AdamSPD

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from count_train_dataset.synthetic_dataloader import create_clip_dataloader

class DistributedLogger:
    def __init__(self, local_rank):
        self.local_rank = local_rank
        self.step_timestamps = {}
    
    def log_step(self, step_name, extra_info=""):
        timestamp = time.time()
        if self.local_rank == 0:
            print(f"[Rank {self.local_rank}] {step_name}: {extra_info} at {timestamp}")
        self.step_timestamps[step_name] = timestamp
        
        # Force flush stdout to ensure logs are written
        sys.stdout.flush()

class DistributedCLIPFineTuner:
    def __init__(self, config: CLIPFineTuneConfig, local_rank: int, checkpoint_path=None):
        self.config = config
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.best_loss = float('inf')
        
        # Initialize the process group
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        if local_rank == 0:
            print(f"Loading CLIP model {config.clip_model}...")
        self.model = CLIPModel.from_pretrained(config.clip_model).to(self.device)
        
        # Wrap model in DDP
        self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
        self.logger = DistributedLogger(local_rank)
        
        # Updated loss function initialization
        if config.loss_type == "sparc":
            self.criterion = SPARCLoss(config).to(self.device)
            if local_rank == 0:
                print("Using SPARC loss")
        elif config.loss_type == "count":
            self.criterion = CLIPCountLoss(temperature=0.07, count_alpha=config.count_alpha).to(self.device)
            if local_rank == 0:
                print(f"Using CLIP+Count loss with alpha={config.count_alpha}")
        else:
            self.criterion = CustomCLIPLoss().to(self.device)
            if local_rank == 0:
                print("Using standard CLIP loss")

        self.optimizer = self.configure_optimizer()
        if local_rank == 0:
            print(f"Using optimizer: {type(self.optimizer).__name__}")

        self.global_step = 0

        if local_rank == 0:
            self.checkpoint_dir = os.path.join("/storage/ice1/9/3/kkundurthy3/checkpoints", config.experiment_name)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.scaler = GradScaler('cuda') if config.use_amp else None

        if checkpoint_path and local_rank == 0:
            self.load_checkpoint(checkpoint_path)
            print(f"Resumed from checkpoint: {checkpoint_path}")
        
        if local_rank == 0:
            print(f"Model loaded and initialized on {self.device}")

    def configure_optimizer(self) -> torch.optim.Optimizer:
        # TODO move this out of fintuner class to optimizer file
        decay_params = []
        no_decay_params = []
        
        # Change this line to use model.module
        for name, param in self.model.module.named_parameters():
            if "ln" in name or "bn" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optimizer_params = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        if self.config.optimizer_type == "adamspd":
            # Change this line to use model.module
            params_to_opt = [p for p in self.model.module.parameters() if p.requires_grad]
            
            # Cache pretrained model weights
            params_anchor = copy.deepcopy(params_to_opt)
            
            optimizer_params = {
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "betas": self.config.betas,
                "eps": self.config.eps,
                "amsgrad": self.config.amsgrad
            }
            
            param_group = [{
                'params': params_to_opt,
                'pre': params_anchor
            }]
            
            return AdamSPD(param_group, **optimizer_params)
        else:
            return torch.optim.AdamW(optimizer_params, lr=self.config.lr)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(batch) == 3:
            images, texts, count_features = batch
            count_features = count_features.to(self.device)
        else:
            images, texts = batch
        images = images.to(self.device)
        texts = texts.to(self.device)

        # if self.local_rank == 0:
        #     print("\nParameters and their requires_grad status:")
        #     for name, param in self.model.named_parameters():
        #         print(f"{name}: requires_grad={param.requires_grad}")

        pad_token_id = self.model.module.config.text_config.pad_token_id
        language_mask = torch.ne(texts, pad_token_id).to(self.device)
        language_mask = language_mask.bool()
        
        if self.config.use_amp:
            with autocast('cuda', enabled=self.config.use_amp):
                outputs = self.model(pixel_values=images, input_ids=texts)
                
                if self.config.loss_type == "sparc":
                    v_patch_embed = outputs.vision_model_output.last_hidden_state
                    v_patch_embed = self.model.module.visual_projection(v_patch_embed)
                    l_token_embed = outputs.text_model_output.last_hidden_state
                    l_token_embed = self.model.module.text_projection(l_token_embed)
                    losses = self.criterion(v_patch_embed, l_token_embed, language_mask)
                elif self.config.loss_type == "count":
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
                    losses = self.criterion(image_features, text_features, count_features)
                else:
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
                    losses = self.criterion(image_features, text_features)

                losses = {k: v / self.config.gradient_accumulation_steps for k, v in losses.items()}
                
            self.scaler.scale(losses["total_loss"]).backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            outputs = self.model(pixel_values=images, input_ids=texts)
            
            if self.config.loss_type == "sparc":
                v_patch_embed = outputs.vision_model_output.last_hidden_state
                v_patch_embed = self.model.module.visual_projection(v_patch_embed)
                l_token_embed = outputs.text_model_output.last_hidden_state
                l_token_embed = self.model.module.text_projection(l_token_embed)
                losses = self.criterion(v_patch_embed, l_token_embed, language_mask)
            elif self.config.loss_type == "count":
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                losses = self.criterion(image_features, text_features, count_features)
            else:
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                losses = self.criterion(image_features, text_features)

            losses = {k: v / self.config.gradient_accumulation_steps for k, v in losses.items()}
            
            losses["total_loss"].backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        self.global_step += 1
        return losses

    def train(self, train_dataloader: DataLoader, num_epochs: int):
        self.model.train()
        
        for epoch in range(num_epochs):
            self.logger.log_step(f"epoch_{epoch}_start")
            train_dataloader.sampler.set_epoch(epoch)
            epoch_losses = []
            
            # Only show progress bar on main process
            if self.local_rank == 0:
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            else:
                progress_bar = train_dataloader
            
            self.optimizer.zero_grad()
            
            for batch in progress_bar:
                losses = self.training_step(batch)
                epoch_losses.append(losses["total_loss"].item() * self.config.gradient_accumulation_steps)
                
                if self.local_rank == 0:
                    progress_bar.set_postfix({
                        'loss': f'{losses["total_loss"].item() * self.config.gradient_accumulation_steps:.4f}',
                        'avg_loss': f'{sum(epoch_losses) / len(epoch_losses):.4f}'
                    })
            
            # Calculate local mean loss
            local_mean_loss = torch.tensor(sum(epoch_losses) / len(epoch_losses), 
                                        device=self.device, dtype=torch.float32)
            
            # Gather mean losses from all processes
            gathered_losses = [torch.zeros_like(local_mean_loss, device=self.device) 
                            for _ in range(dist.get_world_size())]
            
            self.logger.log_step(f"epoch_{epoch}_before_loss_gather")
            torch.distributed.barrier()
            self.logger.log_step(f"epoch_{epoch}_after_barrier")
            
            try:
                dist.all_gather(gathered_losses, local_mean_loss)
                self.logger.log_step(f"epoch_{epoch}_after_loss_gather")
                
                if self.local_rank == 0:
                    # Calculate global average loss
                    global_mean_loss = torch.stack(gathered_losses).mean().item()
                    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {global_mean_loss:.4f}")
                    
                    if global_mean_loss < self.best_loss:
                        self.best_loss = global_mean_loss
                        print(f"New best loss: {self.best_loss:.4f}")
                        self.save_checkpoint('best.pt')

                    if (epoch + 1) % 5 == 0:
                        self.save_checkpoint(f'epoch_{epoch + 1}.pt')
            
            except Exception as e:
                self.logger.log_step(f"epoch_{epoch}_loss_gather_failed", str(e))
                # Continue training even if gathering fails
                torch.distributed.barrier()
                pass
            
            # Make sure all processes are synced before next epoch
            torch.distributed.barrier()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model, optimizer, and training state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load other training state
        self.global_step = checkpoint['global_step']
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
        
        # Load scaler state if it exists
        if self.config.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"Loaded checkpoint from step {self.global_step}")
        
        # Verify config matches
        saved_config = checkpoint['config']
        for key, value in saved_config.items():
            if hasattr(self.config, key) and getattr(self.config, key) != value:
                print(f"Warning: Current config {key}={getattr(self.config, key)} "
                      f"differs from checkpoint config {key}={value}")

    def save_checkpoint(self, filename: str):
        """Save model, optimizer, and training state to checkpoint."""
        # First synchronize all processes
        torch.distributed.barrier()
        
        if self.local_rank == 0:  # Save only on main process
            try:
                path = os.path.join(self.checkpoint_dir, filename)
                config_dict = self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
                
                # Make sure all tensors are on CPU before saving
                checkpoint = {
                    'model_state_dict': {k: v.cpu() for k, v in self.model.module.state_dict().items()},
                    'optimizer_state_dict': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                        for k, v in self.optimizer.state_dict().items()},
                    'config': config_dict,
                    'global_step': self.global_step,
                    'best_loss': self.best_loss
                }
                
                if self.config.use_amp:
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                
                # Save with a temporary file first
                tmp_path = path + '.tmp'
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, path)  # Atomic operation
                
                print(f"Saved checkpoint to {path}")
            except Exception as e:
                print(f"Error saving checkpoint: {str(e)}")
                raise e
        
        # Wait for the saving to complete before continuing
        torch.distributed.barrier()

def main():
    parser = argparse.ArgumentParser(description='Distributed Finetune CLIP model')
    parser.add_argument('--exp_name', type=str, default="clip_coco_finetune",
                        help='experiment name')
    parser.add_argument('--loss_type', type=str, default="clip", 
                        choices=['clip', 'sparc', 'count'])
    parser.add_argument('--optimizer', type=str, default="adamw", 
                        choices=['adamw', 'adamspd'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--save', type=str, help='path to save checkpoint to')
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model_name = "openai/clip-vit-base-patch32"
    config = CLIPFineTuneConfig(
        lr=2e-5,
        batch_size=32,
        max_grad_norm=1.0,
        warmup_steps=100,
        weight_decay=0.1,
        use_amp=True,
        clip_model=model_name,
        experiment_name=args.exp_name,
        gradient_accumulation_steps=4,
        loss_type=args.loss_type,
        similarity_threshold=0.5,
        global_loss_weight=1.0,
        local_loss_weight=1.0,
        inverse_temperature=0.07,
        optimizer_type=args.optimizer,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        count_alpha=1.0,  # Added for count loss
    )

    annotations_file = "/storage/ice1/9/3/kkundurthy3/synthetic_dataset/synthetic_annotations.json"
    image_dir = "/storage/ice1/9/3/kkundurthy3/synthetic_dataset/"
    
    dataloader, _ = create_clip_dataloader(
        annotations_file=annotations_file,
        image_dir=image_dir,
        batch_size=config.batch_size,
        num_workers=4,
        distributed=True,
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        seed=42
    )

    trainer = DistributedCLIPFineTuner(
        config, 
        local_rank,
        checkpoint_path=args.resume if args.resume and local_rank == 0 else None
    )
    trainer.train(dataloader, num_epochs=args.epochs)

    dist.destroy_process_group()

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()