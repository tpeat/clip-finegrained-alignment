import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from transformers import CLIPModel
from tqdm import tqdm
import random
import numpy as np
import os
import copy
import argparse

from dummy_data import create_coco_dataloaders
from losses import CustomCLIPLoss, SPARCLoss
from config import CLIPFineTuneConfig
from optimizers import AdamSPD

class CLIPFineTuner:
    def __init__(self, config: CLIPFineTuneConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading CLIP model {config.clip_model}...")
        self.model = CLIPModel.from_pretrained(config.clip_model).to(self.device)
        
        if config.loss_type == "sparc":
            self.criterion = SPARCLoss(config).to(self.device)
            print("Using SPARC loss")
        else:
            self.criterion = CustomCLIPLoss().to(self.device)
            print("Using standard CLIP loss")

        self.optimizer = self.configure_optimizer()
        print(f"Using optimizer: {type(self.optimizer).__name__}")

        self.global_step = 0 

        self.checkpoint_dir = os.path.join("checkpoints", config.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.scaler = GradScaler() if config.use_amp else None
        
        print(f"Model loaded and initialized on {self.device}")
        
    def configure_optimizer(self) -> torch.optim.Optimizer:
        # TODO move this out of fintuner class to optimizer file
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
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
            # Get trainable parameters
            params_to_opt = [p for p in self.model.parameters() if p.requires_grad]
            
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
        images, texts = batch
        images = images.to(self.device)
        texts = texts.to(self.device)

        pad_token_id = self.model.config.text_config.pad_token_id
        language_mask = torch.ne(texts, pad_token_id).to(self.device)  # [B, T]
        language_mask = language_mask.bool() # we also do this in spd loss
        
        # mp
        if self.config.use_amp:
            with autocast():
                outputs = self.model(pixel_values=images, input_ids=texts)
                
                if self.config.loss_type == "sparc":
                    # project all hidden features to same dim space
                    v_patch_embed = outputs.vision_model_output.last_hidden_state
                    v_patch_embed = self.model.visual_projection(v_patch_embed)
                    l_token_embed = outputs.text_model_output.last_hidden_state
                    l_token_embed = self.model.text_projection(l_token_embed)
  
                    losses = self.criterion(
                        v_patch_embed,
                        l_token_embed,
                        language_mask
                    )
                else:
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
                    losses = self.criterion(image_features, text_features)

                # scale
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
                v_patch_embed = self.model.visual_projection(v_patch_embed)
                l_token_embed = outputs.text_model_output.last_hidden_state
                l_token_embed = self.model.text_projection(l_token_embed)
                losses = self.criterion(
                    v_patch_embed,
                    l_token_embed,
                    language_mask
                )
            else:
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                losses = self.criterion(image_features, text_features)

            # must be scaled by number of steps already taken
            losses = {k: v / self.config.gradient_accumulation_steps for k, v in losses.items()}
            
            losses["total_loss"].backward()
            
            # makes actual gradient steps = batch_size * accum_steps
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        self.global_step += 1
        return losses

    def train(self, train_dataloader: DataLoader, num_epochs: int):
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            self.optimizer.zero_grad()
            
            for batch in progress_bar:
                losses = self.training_step(batch)
                epoch_losses.append(losses["total_loss"].item() * self.config.gradient_accumulation_steps)  # Unscale loss for logging
                
                progress_bar.set_postfix({
                    'loss': f'{losses["total_loss"].item() * self.config.gradient_accumulation_steps:.4f}',
                    'avg_loss': f'{sum(epoch_losses) / len(epoch_losses):.4f}'
                })
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint('best.pt')

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
        }, path)
        print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description='Finetune CLIP model')
    parser.add_argument('--exp_name', type=str, default="clip_coco_finetune",
                        help='experiment name')
    parser.add_argument('--loss_type', type=str, default="clip", choices=['clip', 'sparc'],
                        help='loss function to use (clip or sparc)')
    parser.add_argument('--optimizer', type=str, default="adamw", choices=['adamw', 'adamspd'],
                        help='optimizer to use (adamw or adamspd)')
    args = parser.parse_args()

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
        # for sparc
        loss_type=args.loss_type,
        similarity_threshold=0.5,
        global_loss_weight=1.0,
        local_loss_weight=1.0,
        inverse_temperature=0.07,
        # optimizer
        optimizer_type=args.optimizer,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )

    dataloader = create_coco_dataloaders(
        root_dir='../dataset/coco/train2017',
        ann_file='../dataset/coco/annotations/captions_train2017.json',
        batch_size=config.batch_size,
        model_name=model_name,
        max_samples=1000
    )

    finetuner = CLIPFineTuner(config)
    finetuner.train(dataloader, num_epochs=5)

if __name__ == "__main__":
    main()