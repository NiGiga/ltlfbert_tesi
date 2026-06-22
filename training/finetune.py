"""
LTLfBERT - Fine-tuning su Task B (Conformance Detection)
Confronta: (A) CodeBERT baseline vs (B) LTLfBERT pre-addestrato.
"""

import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ltlfbert import LTLfBERT, ConformanceClassifier


#  DATASET

class ConformanceDataset(Dataset):
    """Dataset completo (label 0 e 1) per Task B."""
    def __init__(self, json_path, tokenizer, max_len=128):
        with open(json_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        f_enc = self.tokenizer(item["formula"], padding="max_length", truncation=True,
                               max_length=self.max_len, return_tensors="pt")
        t_enc = self.tokenizer(item["trace"],   padding="max_length", truncation=True,
                               max_length=self.max_len, return_tensors="pt")
        return {
            "f_ids":  f_enc["input_ids"].squeeze(0),
            "f_mask": f_enc["attention_mask"].squeeze(0),
            "t_ids":  t_enc["input_ids"].squeeze(0),
            "t_mask": t_enc["attention_mask"].squeeze(0),
            "label":  torch.tensor(item["label"], dtype=torch.long),
        }


#  VALUTAZIONE

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                batch["f_ids"].to(device), batch["f_mask"].to(device),
                batch["t_ids"].to(device), batch["t_mask"].to(device))
            preds = logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


#  TRAINING

def finetune_conformance(
    data_dir,
    checkpoint_path=None,  # None = baseline (random init)
    output_dir="ltlfbert/results",
    run_name="baseline",
    epochs=5,
    batch_size=32,
    lr=3e-5,
    freeze_backbone=False,
    use_lora=False,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # Backbone: se checkpoint_path None, pesi random (baseline)
    backbone = LTLfBERT(use_lora=use_lora)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)

        # Carica solo i pesi che matchano — ignora quelli extra
        model_state = backbone.state_dict()
        matched = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        missing = [k for k in model_state if k not in matched]
        unexpected = [k for k in state if k not in model_state]

        backbone.load_state_dict(matched, strict=False)
        print(f"✅ Caricati {len(matched)} pesi | Missing: {len(missing)} | Unexpected: {len(unexpected)}")
        if missing:
            print(f"   Pesi non caricati (prime 3): {missing[:3]}")
    else:
        print("Baseline: pesi CodeBERT originali (senza pre-training LTLf)")

    model = ConformanceClassifier(backbone, freeze_backbone=freeze_backbone).to(device)

    train_dl = DataLoader(ConformanceDataset(os.path.join(data_dir, "train.json"), tokenizer),
                          batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(ConformanceDataset(os.path.join(data_dir, "val.json"), tokenizer),
                          batch_size=batch_size, num_workers=2)
    test_dl  = DataLoader(ConformanceDataset(os.path.join(data_dir, "test.json"), tokenizer),
                          batch_size=batch_size, num_workers=2)
    ood_dl   = DataLoader(ConformanceDataset(os.path.join(data_dir, "ood.json"), tokenizer),
                          batch_size=batch_size, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    # Separa i parametri del backbone da quelli della testa di classificazione
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]

    if freeze_backbone:
        optimizer = optim.AdamW(head_params, lr=lr, weight_decay=0.01)
        print("❄️ Backbone congelato. Alleno solo la projection head.")
    else:
        # LR differenziato: il backbone impara 10 volte piÃ¹ lentamente della head
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': head_params, 'lr': lr}
        ], weight_decay=0.01)
        print(f"🔥Scongelato. LR Backbone: {lr * 0.1} | LR Head: {lr}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        # Prima epoch: congela il backbone, allena solo la testa
        if epoch == 1 and checkpoint_path:
            for p in model.backbone.parameters():
                p.requires_grad_(False)
        elif epoch == 2 and checkpoint_path:
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            print("🔓 Backbone scongelato dall'epoch 2")
        model.train()
        # Attiva gradient checkpointing quando il backbone è scongelato
        if use_lora and (epoch >= 2 or not checkpoint_path):
            model.backbone.encoder.gradient_checkpointing_enable()  # ← aggiunta
        for batch in train_dl:
            logits = model(batch["f_ids"].to(device), batch["f_mask"].to(device),
                           batch["t_ids"].to(device), batch["t_mask"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        val_acc, val_f1 = evaluate(model, val_dl, device)
        print(f"[{run_name}] Epoch {epoch:2d} | val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")
        history.append({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1})

        val_acc, val_f1 = evaluate(model, val_dl, device)
        if val_f1 > best_val_f1:  # monitora F1
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, f"{run_name}_best.pt"))

    # Carica il miglior modello per il test finale
    model.load_state_dict(torch.load(os.path.join(output_dir, f"{run_name}_best.pt"), map_location=device))
    test_acc, test_f1 = evaluate(model, test_dl, device)
    ood_acc,  ood_f1  = evaluate(model, ood_dl,  device)

    results = {
        "run": run_name,
        "test_acc": test_acc, "test_f1": test_f1,
        "ood_acc":  ood_acc,  "ood_f1":  ood_f1,
        "history": history,
    }
    with open(os.path.join(output_dir, f"{run_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{run_name}] TEST: acc={test_acc:.4f}  f1={test_f1:.4f}")
    print(f"[{run_name}] OOD: acc={ood_acc:.4f}  f1={ood_f1:.4f}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="ltlfbert/data")
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--output-dir",  default="ltlfbert/results")
    p.add_argument("--run-name",    default="baseline")
    p.add_argument("--epochs",      type=int, default=5)
    p.add_argument("--lr",          type=float, default=1e-4) # LR della head
    p.add_argument("--freeze",      action="store_true", help="Congela il backbone")
    args = p.parse_args()

    finetune_conformance(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        lr=args.lr,
        freeze_backbone=args.freeze, # Passa l'argomento qui
    )