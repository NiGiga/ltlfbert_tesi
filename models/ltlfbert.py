"""
LTLfBERT - Architettura del Modello
Due encoder CodeBERT + teste di proiezione + NT-Xent loss contrastiva.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ProjectionHead(nn.Module):
    """MLP a 2 strati con LayerNorm e GELU, proietta l'embedding condiviso -> dim_out."""
    def __init__(self, dim_in=768, dim_hidden=512, dim_out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.LayerNorm(dim_hidden), # <--- RIPRISTINATA
            nn.GELU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)  # L2-norm


class NTXentLoss(nn.Module):
    """
    NT-Xent su mini-batch: per N coppie positive, usa N^2-N come negativi impliciti.
    Simmetrica (formula->traccia + traccia->formula).
    """
    def __init__(self, init_temp=0.07, learnable_temp=False):
        super().__init__()
        self.log_temp = nn.Parameter(
            torch.tensor(math.log(init_temp)),
            requires_grad=learnable_temp
        )

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(0.03, 0.20)

    def forward(self, z_formula, z_trace):
        sim = torch.mm(z_formula, z_trace.T) / self.temperature
        labels = torch.arange(z_formula.size(0), device=z_formula.device)
        loss_f2t = F.cross_entropy(sim, labels)
        loss_t2f = F.cross_entropy(sim.T, labels)
        return (loss_f2t + loss_t2f) / 2


class LTLfBERT(nn.Module):
    """
    Rete Siamese: UN SOLO encoder CodeBERT per formule e tracce (Shared Weights).
    Due ProjectionHead separate per proiettare nello spazio condiviso.
    """
    def __init__(self, use_lora=False, lora_rank=8):
        super().__init__()
        from transformers import AutoModel
        model_name = "microsoft/codebert-base"

        if use_lora:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                lora_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=16,
                    target_modules="all-linear",
                    lora_dropout=0.1,
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                base_model = AutoModel.from_pretrained(model_name)
                # UN SOLO ENCODER
                self.encoder = get_peft_model(base_model, lora_cfg)
                # --- ATTIVA GRADIENT CHECKPOINTING ---
                # self.encoder.gradient_checkpointing_enable() # con batch size 64, risparmio 30-40% vram ma 20% + tempo

                # Sblocco manuale degli embeddings per insegnare l'alfabeto
                for name, param in self.encoder.named_parameters():
                    if "embeddings" in name:
                        param.requires_grad = True

                print("LoRA attivato (Rete Siamese - Pesi Condivisi)")
            except ImportError:
                print("peft non trovato, uso full fine-tuning")
                self.encoder = AutoModel.from_pretrained(model_name)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size
        self.proj_phi = ProjectionHead(hidden, 512, 512)
        self.proj_trace = ProjectionHead(hidden, 512, 512)

        self.aux_classifier = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    def _encode(self, proj, input_ids, attention_mask):
        # Usa l'encoder condiviso
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Mean Pooling
        token_embeddings = out.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        return proj(mean_pooled)

    def encode_formula(self, input_ids, attention_mask):
        return self._encode(self.proj_phi, input_ids, attention_mask)

    def encode_trace(self, input_ids, attention_mask):
        return self._encode(self.proj_trace, input_ids, attention_mask)

    def forward(self, f_ids, f_mask, t_ids, t_mask, return_logits=False):
        z_f = self.encode_formula(f_ids, f_mask)
        z_t = self.encode_trace(t_ids, t_mask)
        if not return_logits:
            return z_f, z_t
        # Usa entrambi z_f e z_t per l'aux classifier
        diff = torch.abs(z_f - z_t)
        combined = torch.cat([z_f, z_t, diff], dim=-1)  # 512+512+512=1536
        return z_f, z_t, self.aux_classifier(combined)



class ConformanceClassifier(nn.Module):
    def __init__(self, backbone: LTLfBERT, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, f_ids, f_mask, t_ids, t_mask):
        z_f, z_t = self.backbone(f_ids, f_mask, t_ids, t_mask)
        diff = torch.abs(z_f - z_t)
        features = torch.cat([z_f, z_t, diff], dim=-1)
        return self.classifier(features)


class SatisfiabilityClassifier(nn.Module):
    def __init__(self, backbone: LTLfBERT, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            # Aggiornato per riflettere il nuovo nome dell'encoder
            for p in backbone.encoder.parameters():
                p.requires_grad_(False)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, f_ids, f_mask):
        z_f = self.backbone.encode_formula(f_ids, f_mask)
        return self.classifier(z_f)