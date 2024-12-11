import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Tuple, Optional
from config import CLIPFineTuneConfig

class CustomCLIPLoss(nn.Module):
    """Warmup to ensure we can train on finetune on custom loss"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor,
                custom_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        logits = (image_features @ text_features.t()) / self.temperature
        
        # Ground truth is diagonal
        labels = torch.arange(len(image_features), device=image_features.device)
        
        # Calculate standard CLIP loss
        loss_i = self.cross_entropy(logits, labels)
        loss_t = self.cross_entropy(logits.t(), labels)
        clip_loss = (loss_i + loss_t) / 2.0
        
        losses = {
            "clip_loss": clip_loss,
            "total_loss": clip_loss
        }
        
        return losses


class CLIPCountLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, count_alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        self.count_alpha = count_alpha
        
    def count_loss(self, ei: torch.Tensor, ek: torch.Tensor, 
                   counts: torch.Tensor) -> torch.Tensor:
        device = ei.device
        batch_size = ei.size(0)
        group_size = counts.size(0) // batch_size  # Number of captions per image (1 positive + negatives)
        
        # Convert to float64 for better numerical stability
        ei = ei.to(device, dtype=torch.float64)
        ek = ek.to(device, dtype=torch.float64)
        
        if ei.dim() > 2:
            ei = ei.squeeze(1)
        if ek.dim() > 2:
            ek = ek.squeeze(1)
            
        # Normalize embeddings
        ei = ei / ei.norm(dim=-1, keepdim=True)
        ek = ek / ek.norm(dim=-1, keepdim=True)
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        
        # Process each image with its group of captions (positive + negatives)
        for i in range(batch_size):
            # Get current image embedding
            curr_ei = ei[i]
            
            # Get corresponding group of text embeddings and counts
            start_idx = i * group_size
            end_idx = (i + 1) * group_size
            group_ek = ek[start_idx:end_idx]
            group_counts = counts[start_idx:end_idx]
            
            # First embedding in group is positive example (correct count)
            pos_sim = torch.dot(curr_ei, group_ek[0])
            
            # Rest are negative examples (wrong counts)
            neg_sims = torch.matmul(curr_ei, group_ek[1:].t())
            
            # Compute contrastive loss
            numerator = torch.exp(pos_sim / self.temperature)
            denominator = numerator + torch.sum(torch.exp(neg_sims / self.temperature))
            loss += -torch.log(numerator / denominator)
            
        return loss / batch_size

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor,
                count_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        device = image_features.device
        batch_size = image_features.size(0)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Get expanded batch size (with templates)
        expanded_batch_size = text_features.size(0)
        num_templates = expanded_batch_size // batch_size
        
        # Repeat image features to match text features size
        image_features = image_features.repeat_interleave(num_templates, dim=0)
        
        # Create labels for the expanded batch
        labels = torch.arange(expanded_batch_size, device=device)
        
        # Compute CLIP loss with expanded features
        logits = (image_features @ text_features.t()) / self.temperature
        
        loss_i = self.cross_entropy(logits, labels)
        loss_t = self.cross_entropy(logits.t(), labels)
        clip_loss = (loss_i + loss_t) / 2.0
        
        # Count loss calculation
        count_loss = torch.tensor(0.0, device=device)
        if count_features is not None:
            count_loss = self.count_loss(
                image_features,  # Use expanded image features
                text_features,
                count_features
            ) * self.count_alpha
        
        total_loss = clip_loss + count_loss
        
        return {
            "clip_loss": clip_loss,
            "count_loss": count_loss,
            "total_loss": total_loss
        }


class SPARCLoss(nn.Module):
    """Implementation based on https://arxiv.org/abs/2401.09865"""
    def __init__(self, config):
        super().__init__()
        self.similarity_threshold = config.similarity_threshold
        self.global_loss_weight = config.global_loss_weight
        self.local_loss_weight = config.local_loss_weight
        self.inverse_temperature = config.inverse_temperature

    def pairwise_contrastive_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Input shapes:
        a: [batch_size, embedding_dim]
        b: [batch_size, embedding_dim]
        """
        # Normalize embeddings - paper does this explicitly
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        
        batch_size = a.shape[0]
        # Paper uses eye matrix for labels
        labels = torch.eye(batch_size, device=a.device)
        
        # Calculate similarity matrix
        logits = torch.matmul(a, b.t()) * self.inverse_temperature
        # Use sum reduction to match paper
        loss = F.cross_entropy(logits, torch.arange(batch_size, device=a.device), reduction='sum')
        return loss / batch_size  # Normalize by batch size

    def masked_pairwise_contrastive_loss(self, 
                                       a: torch.Tensor,  # [B, T, D]
                                       b: torch.Tensor,  # [B, T, D]
                                       mask: torch.Tensor  # [B, T]
                                       ) -> torch.Tensor:
        batch_size, seq_len = a.shape[0], a.shape[1]
        
        # Normalize features - paper does this explicitly
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        
        # Create mask for logits [B, T, T]
        mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(1)  # Outer product of masks
        
        # Calculate logits
        logits = torch.bmm(a, b.transpose(1, 2)) * self.inverse_temperature  # [B, T, T]
        
        # Create labels (eye matrix for each batch)
        labels = torch.eye(seq_len, device=a.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Mask logits
        logits = logits.masked_fill(~mask_2d, -float('inf'))
        
        # Calculate loss only for valid positions
        loss = F.cross_entropy(
            logits.view(-1, seq_len),
            torch.arange(seq_len, device=a.device).repeat(batch_size),
            reduction='none'
        ).view(batch_size, seq_len)
        
        # Apply mask and normalize
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def forward(self, 
                v_patch_embed: torch.Tensor,  # [B, P, D]
                l_token_embed: torch.Tensor,  # [B, T, D]
                language_mask: torch.Tensor   # [B, T]
                ) -> Dict[str, torch.Tensor]:
        
        # ---------- GLOBAL LOSS ----------
        # Global feature pooling with proper masking for text
        v_embed = F.normalize(torch.mean(v_patch_embed, dim=1), dim=-1)  # [B, D]
        
        # Masked mean pooling for text
        masked_l_token_embed = l_token_embed * language_mask.unsqueeze(-1)
        token_counts = language_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        l_embed = F.normalize(torch.sum(masked_l_token_embed, dim=1) / token_counts, dim=-1)  # [B, D]
        
        # Global contrastive losses
        loss_vl = self.pairwise_contrastive_loss(v_embed, l_embed)
        loss_lv = self.pairwise_contrastive_loss(l_embed, v_embed)
        global_loss = 0.5 * (loss_vl + loss_lv)
        
        # ---------- LOCAL LOSS ----------
        # Normalize embeddings before similarity calculation
        v_patch_embed_norm = F.normalize(v_patch_embed, dim=-1)
        l_token_embed_norm = F.normalize(l_token_embed, dim=-1)
        
        # Compute similarity
        similarity = torch.einsum('btd,bpd->btp', l_token_embed_norm, v_patch_embed_norm)
        
        # Masked min-max normalization
        similarity_masked = similarity * language_mask.unsqueeze(-1)
        similarity_min = similarity_masked.masked_fill(~language_mask.unsqueeze(-1), float('inf')).min(dim=-1, keepdim=True)[0]
        similarity_max = similarity_masked.masked_fill(~language_mask.unsqueeze(-1), -float('inf')).max(dim=-1, keepdim=True)[0]
        eps = 1e-8
        normalized_similarity = (similarity_masked - similarity_min) / (similarity_max - similarity_min + eps)
        
        # Thresholding
        thresholded_similarity = torch.where(
            normalized_similarity < self.similarity_threshold,
            torch.zeros_like(normalized_similarity),
            normalized_similarity
        )
        
        # Alignment weights
        sum_weights = thresholded_similarity.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        v_align_weights = thresholded_similarity / sum_weights
        
        l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, v_patch_embed)
        
        # then apply contrastive loss lcoally
        loss_vl_local = self.masked_pairwise_contrastive_loss(
            l_grouped_v_patch_embed, l_token_embed, language_mask)
        loss_lv_local = self.masked_pairwise_contrastive_loss(
            l_token_embed, l_grouped_v_patch_embed, language_mask)
        local_loss = 0.5 * (loss_vl_local + loss_lv_local)
        
        total_loss = self.global_loss_weight * global_loss + self.local_loss_weight * local_loss
        
        return {
            "global_loss": global_loss,
            "local_loss": local_loss,
            "total_loss": total_loss,
            "loss_vl": loss_vl,
            "loss_lv": loss_lv,
            "loss_vl_local": loss_vl_local,
            "loss_lv_local": loss_lv_local
        }


class CountLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, alpha = 1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, img_logits, text_logits, ei, ek, ek_cf):
        # contrastive loss
        device = img_logits.device
        ground_truth = torch.arange(len(img_logits), dtype=torch.long, device=device)
        clip_loss = (self.cross_entropy(img_logits, ground_truth) + 
                       self.cross_entropy(text_logits, ground_truth)) / 2

        # Normalize all embeddings
        ei = ei / ei.norm(dim=1, keepdim=True)          # [batch_size, embed_dim]
        ek = ek / ek.norm(dim=1, keepdim=True)          # [batch_size, embed_dim]
        ek_cf = ek_cf / ek_cf.norm(dim=2, keepdim=True) # [batch_size, num_cf, embed_dim]
        
        # Calculate numerator (correct scores) for all examples
        correct_scores = torch.sum(ei * ek, dim=1) / self.temperature # [batch_size]
        numerator = torch.exp(correct_scores)        # [batch_size]
        
        # Calculate denominator using matrix multiplication
        # First reshape ei for broadcasting
        ei_expanded = ei.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Compute all counterfactual scores at once
        cf_scores = torch.sum(ei_expanded * ek_cf, dim=2) / self.temperature # [batch_size, num_cf]
        denominator = torch.sum(torch.exp(cf_scores), dim=1)  # [batch_size]
        
        # Compute final loss
        losses = -torch.log(numerator / denominator)
        
        count_loss = losses.mean()

        total_loss = clip_loss + self.alpha * count_loss

        return {
            "clip_loss": clip_loss,
            "count_loss": count_loss,
            "total_loss": total_loss
        }