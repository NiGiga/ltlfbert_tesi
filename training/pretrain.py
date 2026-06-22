"""
LTLfBERT - Pre-training Contrastivo
Addestra i due encoder con NT-Xent su coppie (formula, traccia soddisfatta).
"""

import json
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.amp import autocast, GradScaler  # Aggiornato per PyTorch moderno
import torch.nn.functional as F
import random
from collections import defaultdict

# Aggiunge il root del progetto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ltlfbert import LTLfBERT, NTXentLoss


# ─── DATASET ────────────────────────────────────────────────────────────────

class LTLfPairDataset(Dataset):
    """Dataset di coppie positive (formula soddisfatta dalla traccia)."""

    def __init__(self, json_path, tokenizer, max_len=128):
        with open(json_path) as f:
            raw = json.load(f)
        self.data = [d for d in raw if d["label"] == 1]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _formula_complexity(self, formula: str) -> int:
        return len(formula.split())

    def __getitem__(self, idx):
        item = self.data[idx]
        f_enc = self.tokenizer(
            item["formula"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        t_enc = self.tokenizer(
            item["trace"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "f_ids": f_enc["input_ids"].squeeze(0),
            "f_mask": f_enc["attention_mask"].squeeze(0),
            "t_ids": t_enc["input_ids"].squeeze(0),
            "t_mask": t_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(item.get("sat_label", 0), dtype=torch.long),
            "formula_str": item["formula"],
            "complexity": self._formula_complexity(item["formula"]),
        }

class HardBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        groups = defaultdict(list)
        for idx in range(len(dataset)):
            c = dataset[idx]["complexity"]
            groups[c].append(idx)

        self.groups = {k: v[:] for k, v in groups.items()}
        self.keys = sorted(self.groups.keys())

    def __iter__(self):
        keys = self.keys[:]
        random.shuffle(keys)

        pools = {k: v[:] for k, v in self.groups.items()}
        for v in pools.values():
            random.shuffle(v)

        batch = []
        seen_formulas = set()

        while keys:
            progress = False
            for k in keys[:]:
                if not pools[k]:
                    keys.remove(k)
                    continue

                idx = pools[k].pop()
                formula = self.dataset[idx]["formula_str"]

                if formula in seen_formulas:
                    continue

                batch.append(idx)
                seen_formulas.add(formula)
                progress = True

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    seen_formulas.clear()
                    progress = False

            if not progress and batch:
                if not self.drop_last:
                    yield batch
                batch = []
                seen_formulas.clear()

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

def unique_formula_collate(batch):
    seen = set()
    filtered = []

    for item in batch:
        f = item["formula_str"]
        if f not in seen:
            seen.add(f)
            filtered.append(item)

    # fallback: se il filtro svuota troppo il batch, usa almeno i primi 2 esempi
    if len(filtered) < 2:
        filtered = batch[:2]

    return {
        "f_ids": torch.stack([x["f_ids"] for x in filtered]),
        "f_mask": torch.stack([x["f_mask"] for x in filtered]),
        "t_ids": torch.stack([x["t_ids"] for x in filtered]),
        "t_mask": torch.stack([x["t_mask"] for x in filtered]),
        "label": torch.stack([x["label"] for x in filtered]),
        "formula_str": [x["formula_str"] for x in filtered],
    }

# ─── TRAINING LOOP ──────────────────────────────────────────────────────────

def pretrain(
        data_dir,
        output_dir,
        epochs=10,
        batch_size=32,
        lr=2e-5,
        use_lora=True,
        device=None,
):
    ACCUM_STEPS = 2
    align_weight = 0.05

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = LTLfBERT(use_lora=use_lora).to(device)
    criterion = NTXentLoss(init_temp=0.07, learnable_temp=True).to(device)

    train_ds = LTLfPairDataset(os.path.join(data_dir, "train.json"), tokenizer)
    val_ds = LTLfPairDataset(os.path.join(data_dir, "val.json"), tokenizer)

    train_sampler = HardBatchSampler(train_ds, batch_size=batch_size, drop_last=True)

    train_dl = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=unique_formula_collate,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=unique_formula_collate,
    )

    # Separiamo i parametri
    lora_params = []
    proj_params = []
    other_params = list(criterion.parameters())

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora" in name.lower():
                lora_params.append(param)
            elif "proj" in name.lower():
                proj_params.append(param)
            else:
                other_params.append(param)

    if use_lora:
        optimizer = optim.AdamW([
            {'params': lora_params, 'lr': 1e-4},
            {'params': proj_params, 'lr': 1e-4},
            {'params': other_params, 'lr': 1e-5}
        ], weight_decay=0.01)
        print("🔥 Optimizer Config: LoRA LR=1e-4, Proj LR=1e-4, Other LR=1e-5")
    else:
        optimizer = optim.AdamW([
            {'params': other_params, 'lr': lr},
            {'params': proj_params, 'lr': lr * 5}
        ], weight_decay=0.01)
        print(f"🔥 Optimizer Config: Backbone LR={lr}, Proj LR={lr * 5}")

    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    history = []
    scaler = GradScaler('cuda')

    patience = 5
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        criterion.train()
        if use_lora:
            model.encoder.gradient_checkpointing_enable()
        train_loss = 0.0
        train_contrastive = 0.0
        train_cls = 0.0
        train_align = 0.0


        for step, batch in enumerate(train_dl):
            f_ids = batch["f_ids"].to(device)
            f_mask = batch["f_mask"].to(device)
            t_ids = batch["t_ids"].to(device)
            t_mask = batch["t_mask"].to(device)

            with autocast('cuda'):
                z_f, z_t = model(f_ids, f_mask, t_ids, t_mask)
                contrastive_loss = criterion(z_f, z_t)
                align_loss = 1.0 - F.cosine_similarity(z_f, z_t, dim=-1).mean()
                loss = (contrastive_loss + align_weight * align_loss) / ACCUM_STEPS  # ← dividi

            scaler.scale(loss).backward()  # accumula i gradienti

            if (step + 1) % ACCUM_STEPS == 0:  # ogni 2 step, aggiorna
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # ← zero_grad DOPO l'update, non prima del forward
                scheduler.step()

            train_loss += loss.item() * ACCUM_STEPS
            train_contrastive += contrastive_loss.item()
            train_align += align_loss.item()


        train_loss /= len(train_dl)
        train_contrastive /= len(train_dl)
        train_cls /= len(train_dl)
        train_align /= len(train_dl)

        model.eval()
        if use_lora:
            model.encoder.gradient_checkpointing_enable()
        criterion.eval()
        val_loss = 0.0
        val_contrastive = 0.0
        val_cls = 0.0
        val_align = 0.0

        with torch.no_grad():
            for batch in val_dl:
                f_ids = batch["f_ids"].to(device)
                f_mask = batch["f_mask"].to(device)
                t_ids = batch["t_ids"].to(device)
                t_mask = batch["t_mask"].to(device)
                labels = batch["label"].to(device)

                z_f, z_t = model(f_ids, f_mask, t_ids, t_mask)
                contrastive_loss = criterion(z_f, z_t)
                align_loss = 1.0 - F.cosine_similarity(z_f, z_t, dim=-1).mean()
                batch_val_loss = contrastive_loss + align_weight * align_loss

                if epoch <= 1:
                    cls_weight = 0.0
                elif epoch <= 3:
                    cls_weight = 0.02
                else:
                    cls_weight = 0.05

                val_loss += batch_val_loss.item()
                val_contrastive += contrastive_loss.item()
                val_align += align_loss.item()

        val_loss /= len(val_dl)
        val_contrastive /= len(val_dl)
        val_cls /= len(val_dl)
        val_align /= len(val_dl)

        temp = criterion.temperature.item()
        print(
            f"Epoch {epoch:3d} | "
            f"train={train_loss:.4f} "
            f"(ctr={train_contrastive:.4f}, cls={train_cls:.4f}, align={train_align:.4f}) | "
            f"val={val_loss:.4f} "
            f"(ctr={val_contrastive:.4f}, cls={val_cls:.4f}, align={val_align:.4f}) | "
            f"temp={temp:.4f} | cls_w={cls_weight:.3f} | align_w={align_weight:.3f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_contrastive": train_contrastive,
            "train_cls": train_cls,
            "train_align": train_align,
            "val_loss": val_loss,
            "val_contrastive": val_contrastive,
            "val_cls": val_cls,
            "val_align": val_align,
            "temperature": temp,
            "cls_weight": cls_weight,
            "align_weight": align_weight,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "ltlfbert_best.pt"))
            print(f"  → Checkpoint salvato (val_loss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⚠️ Early stopping counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print("  🛑 Early Stopping attivato!")
                break

    with open(os.path.join(output_dir, "pretrain_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="ltlfbert/data")
    p.add_argument("--output-dir", default="ltlfbert/checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--no-lora", action="store_true")
    args = p.parse_args()

    pretrain(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_lora=not args.no_lora,
    )