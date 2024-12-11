# rework of finetuner
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
import os
import clip
import argparse
import copy

from config import CLIPFineTuneConfig
from losses import CountLoss
from evaluate import evaluate_batch
from optimizers import AdamSPD

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from count_train_dataset.count_dataloader import create_clip_dataloader

class CountFineTuner:
    def __init__(self, config: CLIPFineTuneConfig, checkpoint_path=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading CLIP model {config.clip_model}...")
        self.model, _ = clip.load(config.clip_model, device=self.device, jit=False)
        self.model = self.model.float() # ensure float32
        
        self.criterion = CountLoss(
            temperature=config.inverse_temperature, 
            alpha=config.count_alpha
        ).to(self.device)
        
        print(f"Using Count loss with temperature={config.inverse_temperature}, alpha={config.count_alpha}")

        self.optimizer = self.configure_optimizer()
        print(f"Using optimizer: {type(self.optimizer).__name__}")

        self.global_step = 0
        self.best_loss = float('inf')
        
        self.checkpoint_dir = os.path.join("checkpoints", config.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.eval_dir = os.path.join("eval_results", config.experiment_name)
        os.makedirs(self.eval_dir, exist_ok=True)
        
        self.scaler = GradScaler() if config.use_amp else None

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def configure_optimizer(self):
        if hasattr(self.config, 'optimizer_type') and self.config.optimizer_type == "adamspd":
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
            # og adam
            decay_params = []
            no_decay_params = []
            
            for name, param in self.model.named_parameters():
                if "ln" in name or "bn" in name or "bias" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
                    
            optimizer_params = [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]

            return torch.optim.AdamW(
                optimizer_params, 
                lr=self.config.lr,
                betas=self.config.betas,
                eps=self.config.eps,
                amsgrad=self.config.amsgrad
            )

    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        images = batch['image'].to(self.device)
        texts = batch['text'].to(self.device)
        cf_texts = batch['cf_text'].to(self.device)

        with autocast(enabled=self.config.use_amp):
            encoded_images = self.model.encode_image(images)
            encoded_texts = self.model.encode_text(texts)
            
            all_cf_embeddings = []
            for i in range(len(images)):
                cf_embeddings = self.model.encode_text(cf_texts[i])
                all_cf_embeddings.append(cf_embeddings)
            encoded_all_cf_texts = torch.stack(all_cf_embeddings)
            
            image_features = encoded_images / encoded_images.norm(dim=-1, keepdim=True)
            text_features = encoded_texts / encoded_texts.norm(dim=-1, keepdim=True)

            # TODO: authors of TeachClipTo.. didn't logit scale
            # logit_scale = self.model.logit_scale.exp()
            # logits_per_image = logit_scale * image_features @ text_features.t()
            # logits_per_text = logits_per_image.t()
            logits_per_image, logits_per_text = self.model(images, texts)

            losses = self.criterion(
                logits_per_image, 
                logits_per_text, 
                encoded_images, 
                encoded_texts, 
                encoded_all_cf_texts
            )
            
            losses = {k: v / self.config.gradient_accumulation_steps for k, v in losses.items()}

        if self.config.use_amp:
            self.scaler.scale(losses["total_loss"]).backward()
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            losses["total_loss"].backward()
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        self.global_step += 1
        return losses

    def evaluate_epoch(self, batch, epoch: int):
        """Run evaluation on a single batch after each epoch."""
        self.model.eval()
        with torch.no_grad():
            accuracy, confusion, results = evaluate_batch(
                model=self.model,
                batch=batch,
                device=self.device,
                filename=os.path.join(self.eval_dir, f'confusion_matrix_epoch_{epoch}.png')
            )
        self.model.train()
        
        print(f"\nEpoch {epoch} Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        
        return accuracy, confusion, results

    def train(self, train_dataloader: DataLoader):
        self.model.train()

        # store the first batch as a small val set
        eval_batch = next(iter(train_dataloader))
        accuracy, confusion, results = self.evaluate_epoch(eval_batch, 0)

        num_epochs = self.config.max_epochs
        for epoch in range(num_epochs):
            epoch_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            self.optimizer.zero_grad()
            
            for batch in progress_bar:
                losses = self.training_step(batch)
                epoch_losses.append(losses["total_loss"].item() * self.config.gradient_accumulation_steps)
                
                progress_bar.set_postfix({
                    'total_loss': f'{losses["total_loss"].item() * self.config.gradient_accumulation_steps:.4f}',
                    'clip_loss': f'{losses["clip_loss"].item() * self.config.gradient_accumulation_steps:.4f}',
                    'count_loss': f'{losses["count_loss"].item() * self.config.gradient_accumulation_steps:.4f}'
                })

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            accuracy, confusion, results = self.evaluate_epoch(eval_batch, epoch + 1)
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                print(f"New best loss: {self.best_loss:.4f}")
                self.save_checkpoint('best.pt')

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
            'best_loss': self.best_loss
        }
        
        if self.config.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path, _use_new_zipfile_serialization=True)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.config.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finetune CLIP model')
    parser.add_argument('--exp_name', type=str, default="clip_coco_finetune",
                        help='experiment name')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='max num epochs to train for')
    parser.add_argument('--optimizer', type=str, default="adamw", 
                        choices=['adamw', 'adamspd'],
                        help='optimizer to use (adamw or adamspd)')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    annotations_file = "../count_train_dataset/synthetic_dataset/synthetic_annotations.json"
    image_dir = "../count_train_dataset/"

    config = CLIPFineTuneConfig(
        experiment_name=args.exp_name,
        max_epochs=args.epochs,
        optimizer_type=args.optimizer
    )
    config.print_config()
    
    dataloader, dataset = create_clip_dataloader(
        annotations_file=annotations_file,
        image_dir=image_dir,
        batch_size=config.batch_size,
        num_workers=4
    )

    checkpoint_path = None # "../tmp/clip_final_model.pt"
    finetuner = CountFineTuner(
        config=config,
        checkpoint_path=checkpoint_path
    )
    
    finetuner.train(train_dataloader=dataloader)