"""
LTLfBERT - Script di Valutazione Completo
- Task B: Conformance Detection (accuracy, F1)
- Task C: Retrieval (MRR, Recall@1, Recall@10)
- Analisi latent space: PCA + t-SNE
- Confronto baseline vs LTLfBERT
"""

import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ltlfbert import LTLfBERT
from training.finetune import ConformanceDataset, evaluate


# ─── RETRIEVAL ───────────────────────────────────────────────────────────────

def retrieval_metrics(model, dataset, tokenizer, device, max_items=500):
    """
    Task C: data una formula, trova la traccia più vicina nello spazio latente.
    Calcola MRR e Recall@K su un sottoinsieme (per velocità).
    """
    model.eval()
    # Filtra solo esempi positivi
    pos_items = [d for d in dataset if d["label"] == 1][:max_items]
    neg_items = [d for d in dataset if d["label"] == 0][:max_items]

    def enc_formula(item):
        e = tokenizer(item["formula"], return_tensors="pt",
                      padding="max_length", truncation=True, max_length=128)
        with torch.no_grad():
            return model.encode_formula(e["input_ids"].to(device),
                                        e["attention_mask"].to(device)).cpu().numpy()[0]

    def enc_trace(item):
        e = tokenizer(item["trace"], return_tensors="pt",
                      padding="max_length", truncation=True, max_length=128)
        with torch.no_grad():
            return model.encode_trace(e["input_ids"].to(device),
                                      e["attention_mask"].to(device)).cpu().numpy()[0]

    # Matrice di embedding
    f_embs = np.stack([enc_formula(d) for d in pos_items])  # (N, 256)
    # Pool di tracce: positive + negative mescolate (galleria di retrieval)
    gallery = pos_items + neg_items
    np.random.shuffle(gallery)
    t_embs = np.stack([enc_trace(d) for d in gallery])       # (M, 256)

    # Indici delle tracce positive nella galleria
    pos_traces_set = {d["trace"] for d in pos_items}

    mrr_sum, r1, r10 = 0.0, 0, 0
    for i, f_emb in enumerate(f_embs):
        target_trace = pos_items[i]["trace"]
        sims = t_embs @ f_emb                     # (M,)
        order = np.argsort(-sims)
        for rank, j in enumerate(order, 1):
            if gallery[j]["trace"] == target_trace:
                mrr_sum += 1.0 / rank
                if rank == 1:  r1  += 1
                if rank <= 10: r10 += 1
                break

    n = len(pos_items)
    return {"MRR": mrr_sum/n, "Recall@1": r1/n, "Recall@10": r10/n}


# ─── LATENT SPACE ────────────────────────────────────────────────────────────

def extract_embeddings(model, data, tokenizer, device, max_items=300):
    """Estrae embedding di formula per visualizzazione."""
    model.eval()
    items = data[:max_items]
    embs, labels, depths = [], [], []
    for item in items:
        e = tokenizer(item["formula"], return_tensors="pt",
                      padding="max_length", truncation=True, max_length=128)
        with torch.no_grad():
            z = model.encode_formula(e["input_ids"].to(device),
                                     e["attention_mask"].to(device)).cpu().numpy()[0]
        embs.append(z)
        labels.append(item["label"])
        depths.append(item["depth"])
    return np.array(embs), np.array(labels), np.array(depths)


def plot_latent_space(embs, labels, depths, title, output_path):
    """PCA + t-SNE del latent space, colorato per label e depth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # PCA
    pca = PCA(n_components=2)
    pca_embs = pca.fit_transform(embs)
    colors = ["#e74c3c" if l == 1 else "#3498db" for l in labels]
    axes[0].scatter(pca_embs[:, 0], pca_embs[:, 1], c=colors, alpha=0.6, s=15)
    axes[0].set_title(f"PCA (var. spiegata: {sum(pca.explained_variance_ratio_)*100:.1f}%)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    patch_sat = mpatches.Patch(color="#e74c3c", label="SAT (label=1)")
    patch_unk = mpatches.Patch(color="#3498db", label="UNSAT (label=0)")
    axes[0].legend(handles=[patch_sat, patch_unk], fontsize=8)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
    tsne_embs = tsne.fit_transform(embs)
    axes[1].scatter(tsne_embs[:, 0], tsne_embs[:, 1], c=colors, alpha=0.6, s=15)
    axes[1].set_title("t-SNE")
    axes[1].set_xlabel("Dim 1"); axes[1].set_ylabel("Dim 2")
    axes[1].legend(handles=[patch_sat, patch_unk], fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot salvato: {output_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def full_evaluation(
    data_dir="ltlfbert/data",
    baseline_ckpt=None,         # None = pesi base CodeBERT
    ltlfbert_ckpt=None,         # checkpoint dopo pre-training
    output_dir="ltlfbert/results",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    with open(os.path.join(data_dir, "test.json")) as f:
        test_data = json.load(f)
    with open(os.path.join(data_dir, "ood.json")) as f:
        ood_data = json.load(f)

    report = {}

    for run_name, ckpt_path in [("baseline", baseline_ckpt), ("ltlfbert", ltlfbert_ckpt)]:
        print(f"\n=== Valutazione: {run_name} ===")
        model = LTLfBERT(use_lora=False)
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            # Il checkpoint potrebbe essere del ConformanceClassifier
            if any(k.startswith("backbone.") for k in state):
                new_state = {k.replace("backbone.", ""): v
                             for k, v in state.items() if k.startswith("backbone.")}
                model.load_state_dict(new_state, strict=False)
            else:
                model.load_state_dict(state, strict=False)
        model = model.to(device)

        ret_test = retrieval_metrics(model, test_data, tokenizer, device)
        ret_ood  = retrieval_metrics(model, ood_data,  tokenizer, device)
        print(f"  Retrieval TEST: {ret_test}")
        print(f"  Retrieval OOD:  {ret_ood}")

        embs, labels, depths = extract_embeddings(model, test_data, tokenizer, device)
        plot_latent_space(
            embs, labels, depths,
            title=f"Latent Space – {run_name}",
            output_path=os.path.join(output_dir, f"latent_{run_name}.png")
        )

        report[run_name] = {"retrieval_test": ret_test, "retrieval_ood": ret_ood}

    with open(os.path.join(output_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Confronto finale
    print("\n" + "="*50)
    print("CONFRONTO BASELINE vs LTLfBERT")
    print("="*50)
    for metric in ["MRR", "Recall@1", "Recall@10"]:
        b = report.get("baseline", {}).get("retrieval_test", {}).get(metric, 0)
        l = report.get("ltlfbert", {}).get("retrieval_test", {}).get(metric, 0)
        delta = (l - b) * 100
        print(f"  {metric:12s} | baseline={b:.3f}  ltlfbert={l:.3f}  Δ={delta:+.1f}pp")

    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",       default="ltlfbert/data")
    p.add_argument("--baseline-ckpt",  default=None)
    p.add_argument("--ltlfbert-ckpt",  default=None)
    p.add_argument("--output-dir",     default="ltlfbert/results")
    args = p.parse_args()
    full_evaluation(
        data_dir=args.data_dir,
        baseline_ckpt=args.baseline_ckpt,
        ltlfbert_ckpt=args.ltlfbert_ckpt,
        output_dir=args.output_dir,
    )
