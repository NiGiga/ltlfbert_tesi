#!/usr/bin/env python3
"""
analyze_results.py
Genera curve di loss, grafici di confronto e CSV dai risultati dei run LTLfBERT.

Struttura attesa:
  ltlfbert/
    checkpoints_quick/   ← pretrain_history.json  (lista di dict per epoch)
    checkpoints_10/
    checkpoints_comp/
    results_quick/       ← baseline_results.json, ltlfbert_results.json
    results_10/
    results_comp/

Uso:
    python analyze_results.py
    python analyze_results.py --runs quick 10 comp
    python analyze_results.py --base-dir ltlfbert --out ltlfbert/analysis
"""

import argparse
import csv
import json
import os

import plotly.graph_objects as go

# ── Configurazione ────────────────────────────────────────────────────────────
DEFAULT_RUNS   = ["quick", "10", "comp"]
DEFAULT_LABELS = {"quick": "Quick (1k)", "10": "Mid (10k)", "comp": "Comp (5k)"}

PALETTE_SOLID = ["#60A5FA", "#FBBF24", "#34D399", "#F472B6", "#A78BFA"]
PALETTE_DASH  = ["#93C5FD", "#FDE68A", "#6EE7B7", "#FBCFE8", "#DDD6FE"]

BG    = "#1E1E2E"
GRID  = "#2D2D44"
WHITE = "#FFFFFF"
MUTED = "#AAAACC"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def apply_style(fig, title, xlab, ylab, yrange=None):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=WHITE), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Arial, sans-serif", size=13, color=WHITE),
        xaxis=dict(
            title=dict(text=xlab, font=dict(size=13, color=MUTED)),
            tickfont=dict(size=12, color=MUTED),
            gridcolor=GRID, showgrid=True, zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=ylab, font=dict(size=13, color=MUTED)),
            tickfont=dict(size=12, color=MUTED),
            gridcolor=GRID, showgrid=True, zeroline=False,
        ),

        legend=dict(
            orientation="h",
            yanchor="top",      # era "bottom"
            y=-0.22,            # era 1.08  → sposta sotto l'asse X
            xanchor="center", x=0.5,
            font=dict(size=11, color=WHITE), bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=70, r=30, t=70, b=130),  # era b=60 → più spazio sotto
        # ─────────────────────────────────────────────────────────
        width=900, height=480,
    )
    if yrange:
        fig.update_yaxes(range=yrange)


def save_fig(fig, path, caption, description=""):
    fig.write_image(path)
    with open(path + ".meta.json", "w") as f:
        json.dump({"caption": caption, "description": description}, f)
    print(f"  ✅ {path}")


# ── Plot: pretrain ────────────────────────────────────────────────────────────
def plot_pretrain_total(runs_data, out):
    fig = go.Figure()
    for i, (run, data) in enumerate(runs_data.items()):
        hist = data.get("pretrain_history", [])
        if not hist:
            continue
        label = data["label"]
        ep = [h["epoch"] for h in hist]
        fig.add_trace(go.Scatter(x=ep, y=[h["train_loss"] for h in hist],
            name=f"{label} train", line=dict(color=PALETTE_SOLID[i % len(PALETTE_SOLID)], width=2.5),
            mode="lines+markers", marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=ep, y=[h["val_loss"] for h in hist],
            name=f"{label} val", line=dict(color=PALETTE_DASH[i % len(PALETTE_DASH)], width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=4)))
    apply_style(fig, "Loss totale pretraining", "Epoch", "Loss")
    save_fig(fig, os.path.join(out, "pretrain_loss.png"),
             "Pretrain loss totale per tutti i run",
             "Train (solid) e val (dash) — somma contrastive + align")


def plot_pretrain_contrastive(runs_data, out):
    fig = go.Figure()
    for i, (run, data) in enumerate(runs_data.items()):
        hist = data.get("pretrain_history", [])
        if not hist or "train_contrastive" not in hist[0]:
            continue
        label = data["label"]
        ep = [h["epoch"] for h in hist]
        fig.add_trace(go.Scatter(x=ep, y=[h["train_contrastive"] for h in hist],
            name=f"{label} train", line=dict(color=PALETTE_SOLID[i % len(PALETTE_SOLID)], width=2.5),
            mode="lines+markers", marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=ep, y=[h["val_contrastive"] for h in hist],
            name=f"{label} val", line=dict(color=PALETTE_DASH[i % len(PALETTE_DASH)], width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=4)))
    apply_style(fig, "NT-Xent contrastive loss pretraining", "Epoch", "NT-Xent loss")
    save_fig(fig, os.path.join(out, "pretrain_contrastive_loss.png"),
             "NT-Xent loss pretraining per tutti i run")


def plot_pretrain_align(runs_data, out):
    fig = go.Figure()
    for i, (run, data) in enumerate(runs_data.items()):
        hist = data.get("pretrain_history", [])
        if not hist or "train_align" not in hist[0]:
            continue
        label = data["label"]
        ep = [h["epoch"] for h in hist]
        fig.add_trace(go.Scatter(x=ep, y=[h["train_align"] for h in hist],
            name=f"{label} train", line=dict(color=PALETTE_SOLID[i % len(PALETTE_SOLID)], width=2.5),
            mode="lines+markers", marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=ep, y=[h["val_align"] for h in hist],
            name=f"{label} val", line=dict(color=PALETTE_DASH[i % len(PALETTE_DASH)], width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=4)))
    apply_style(fig, "Align loss pretraining (1 − cosine_sim)", "Epoch", "Align loss")
    save_fig(fig, os.path.join(out, "pretrain_align_loss.png"),
             "Align loss: allineamento tra z_f e z_t nel pretraining")


def plot_temperature(runs_data, out):
    """Curva della temperature learnable durante il pretraining."""
    fig = go.Figure()
    has_data = False
    for i, (run, data) in enumerate(runs_data.items()):
        hist = data.get("pretrain_history", [])
        if not hist or "temperature" not in hist[0]:
            continue
        has_data = True
        label = data["label"]
        ep = [h["epoch"] for h in hist]
        fig.add_trace(go.Scatter(x=ep, y=[h["temperature"] for h in hist],
            name=label, line=dict(color=PALETTE_SOLID[i % len(PALETTE_SOLID)], width=2.5),
            mode="lines+markers", marker=dict(size=5)))
    if not has_data:
        return
    apply_style(fig, "Temperature NT-Xent (learnable)", "Epoch", "Temperature τ")
    save_fig(fig, os.path.join(out, "pretrain_temperature.png"),
             "Evoluzione della temperature learnable durante il pretraining")


# ── Plot: fine-tuning ─────────────────────────────────────────────────────────
def plot_finetune_f1(runs_data, out):
    fig = go.Figure()
    for i, (run, data) in enumerate(runs_data.items()):
        bh = data.get("baseline_results", {}).get("history", [])
        lh = data.get("ltlfbert_results", {}).get("history", [])
        if not lh:
            continue
        label = data["label"]
        ep_l = [h["epoch"] for h in lh]
        fig.add_trace(go.Scatter(x=ep_l, y=[h["val_f1"] for h in lh],
            name=f"{label} LTLfBERT",
            line=dict(color=PALETTE_SOLID[i % len(PALETTE_SOLID)], width=2.5),
            mode="lines+markers", marker=dict(size=5)))
        if bh:
            ep_b = [h["epoch"] for h in bh]
            fig.add_trace(go.Scatter(x=ep_b, y=[h["val_f1"] for h in bh],
                name=f"{label} Baseline",
                line=dict(color=PALETTE_DASH[i % len(PALETTE_DASH)], width=1.8, dash="dash"),
                mode="lines+markers", marker=dict(size=4)))
    apply_style(fig, "Val F1 fine-tuning — LTLfBERT vs Baseline",
                "Epoch", "Val F1 macro", [0.3, 0.85])
    save_fig(fig, os.path.join(out, "finetune_val_f1.png"),
             "Curva val F1 fine-tuning per tutti i run")

def plot_finetune_acc(runs_data, out):
    fig = go.Figure()
    for i, (run, data) in enumerate(runs_data.items()):
        bh = data.get("baseline_results", {}).get("history", [])
        lh = data.get("ltlfbert_results", {}).get("history", [])
        if not lh:
            continue
        label = data["label"]
        ep_l = [h["epoch"] for h in lh]
        fig.add_trace(go.Scatter(x=ep_l, y=[h["val_acc"] for h in lh],
            name=f"{label} LTLfBERT",
            line=dict(color=PALETTE_SOLID[i % len(PALETTE_SOLID)], width=2.5),
            mode="lines+markers", marker=dict(size=5)))
        if bh:
            ep_b = [h["epoch"] for h in bh]
            fig.add_trace(go.Scatter(x=ep_b, y=[h["val_acc"] for h in bh],
                name=f"{label} Baseline",
                line=dict(color=PALETTE_DASH[i % len(PALETTE_DASH)], width=1.8, dash="dash"),
                mode="lines+markers", marker=dict(size=4)))
    apply_style(fig, "Val Accuracy fine-tuning — LTLfBERT vs Baseline",
                "Epoch", "Val Accuracy macro", [0.3, 0.85])
    save_fig(fig, os.path.join(out, "finetune_val_acc.png"),
             "Curva val accuracy fine-tuning per tutti i run")


def plot_comparison_bar(runs_data, out):
    run_labels = [data["label"] for data in runs_data.values()]
    has_baseline = any(data.get("baseline_results") for data in runs_data.values())

    fig = go.Figure()
    if has_baseline:
        fig.add_trace(go.Bar(
            name="Baseline TEST F1", x=run_labels,
            y=[data.get("baseline_results", {}).get("test_f1", 0) for data in runs_data.values()],
            marker_color="#64748B",
            text=[f"{data.get('baseline_results',{}).get('test_f1',0):.3f}" for data in runs_data.values()],
            textposition="outside", textfont=dict(size=12, color=WHITE)))
    fig.add_trace(go.Bar(
        name="LTLfBERT TEST F1", x=run_labels,
        y=[data.get("ltlfbert_results", {}).get("test_f1", 0) for data in runs_data.values()],
        marker_color="#60A5FA",
        text=[f"{data.get('ltlfbert_results',{}).get('test_f1',0):.3f}" for data in runs_data.values()],
        textposition="outside", textfont=dict(size=12, color=WHITE)))
    if has_baseline:
        fig.add_trace(go.Bar(
            name="Baseline OOD F1", x=run_labels,
            y=[data.get("baseline_results", {}).get("ood_f1", 0) for data in runs_data.values()],
            marker_color="#94A3B8",
            text=[f"{data.get('baseline_results',{}).get('ood_f1',0):.3f}" for data in runs_data.values()],
            textposition="outside", textfont=dict(size=12, color=WHITE)))
    fig.add_trace(go.Bar(
        name="LTLfBERT OOD F1", x=run_labels,
        y=[data.get("ltlfbert_results", {}).get("ood_f1", 0) for data in runs_data.values()],
        marker_color="#34D399",
        text=[f"{data.get('ltlfbert_results',{}).get('ood_f1',0):.3f}" for data in runs_data.values()],
        textposition="outside", textfont=dict(size=12, color=WHITE)))
    fig.update_layout(barmode="group")
    apply_style(fig, "TEST & OOD F1 — Baseline vs LTLfBERT", "Run", "F1 macro", [0, 0.92])
    fig.update_traces(cliponaxis=False)
    save_fig(fig, os.path.join(out, "comparison_f1.png"),
             "F1 macro TEST e OOD: Baseline vs LTLfBERT per tutti i run")


# ── CSV ───────────────────────────────────────────────────────────────────────
def csv_pretrain_history(runs_data, out):
    path = os.path.join(out, "pretrain_loss_history.csv")
    fields = ["run", "epoch", "train_loss", "val_loss",
              "train_contrastive", "val_contrastive",
              "train_align", "val_align",
              "train_cls", "val_cls",
              "temperature", "cls_weight", "align_weight"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for run, data in runs_data.items():
            for h in data.get("pretrain_history", []):
                row = {k: h.get(k, "") for k in fields}
                row["run"] = data["label"]
                w.writerow(row)
    print(f"  ✅ {path}")


def csv_finetune_history(runs_data, out):
    path = os.path.join(out, "finetune_val_history.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "model", "epoch", "val_acc", "val_f1"])
        for run, data in runs_data.items():
            label = data["label"]
            for model_key, model_name in [("baseline_results", "baseline"),
                                           ("ltlfbert_results", "ltlfbert")]:
                for h in data.get(model_key, {}).get("history", []):
                    w.writerow([label, model_name,
                                h.get("epoch", ""), h.get("val_acc", ""), h.get("val_f1", "")])
    print(f"  ✅ {path}")


def csv_final_comparison(runs_data, out):
    path = os.path.join(out, "final_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "model", "test_acc", "test_f1", "ood_acc", "ood_f1",
                    "delta_test_acc_pp", "delta_test_f1_pp",
                    "delta_ood_acc_pp", "delta_ood_f1_pp"])
        for run, data in runs_data.items():
            label = data["label"]
            b = data.get("baseline_results", {})
            l = data.get("ltlfbert_results", {})
            if b:
                w.writerow([label, "baseline",
                            b.get("test_acc",""), b.get("test_f1",""),
                            b.get("ood_acc",""),  b.get("ood_f1",""),
                            "", "", "", ""])
            if l:
                w.writerow([label, "ltlfbert",
                            l.get("test_acc",""), l.get("test_f1",""),
                            l.get("ood_acc",""),  l.get("ood_f1",""),
                            round((l.get("test_acc",0) - b.get("test_acc",0)) * 100, 2) if b else "",
                            round((l.get("test_f1",0)  - b.get("test_f1",0))  * 100, 2) if b else "",
                            round((l.get("ood_acc",0)  - b.get("ood_acc",0))  * 100, 2) if b else "",
                            round((l.get("ood_f1",0)   - b.get("ood_f1",0))   * 100, 2) if b else "",
                            ])
    print(f"  ✅ {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analisi risultati LTLfBERT")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                        help="Suffissi dei run, es: quick 10 comp (default: tutti e 3)")
    parser.add_argument("--base-dir", default="ltlfbert",
                        help="Cartella radice del progetto (default: ltlfbert)")
    parser.add_argument("--out", default=None,
                        help="Cartella output (default: <base-dir>/analysis)")
    args = parser.parse_args()

    base = args.base_dir
    out  = args.out or os.path.join(base, "analysis")
    os.makedirs(out, exist_ok=True)

    print(f"📂 Base dir : {base}")
    print(f"📊 Output   : {out}")
    print(f"🔍 Run      : {args.runs}\n")

    runs_data = {}
    for i, run in enumerate(args.runs):
        label = DEFAULT_LABELS.get(run, run)
        ckpt_dir   = os.path.join(base, f"checkpoints_{run}")
        result_dir = os.path.join(base, f"results_{run}")

        entry = {"label": label}

        # ── Pretrain history da checkpoints_<run>/pretrain_history.json ──
        ph_path = os.path.join(ckpt_dir, "pretrain_history.json")
        ph = load_json(ph_path)
        if ph:
            # Supporta sia lista diretta sia dict con chiave "history"
            if isinstance(ph, dict) and "history" in ph:
                ph = ph["history"]
            entry["pretrain_history"] = ph
            print(f"  ✅ Pretrain history: {ph_path} ({len(ph)} epoch)")
        else:
            print(f"  ⚠️  Pretrain history non trovata: {ph_path}")

        # ── Fine-tuning results da results_<run>/ ───────────────────────
        for key in ["baseline_results", "ltlfbert_results"]:
            res = load_json(os.path.join(result_dir, f"{key}.json"))
            if res:
                if "history" not in res:
                    hist = load_json(os.path.join(result_dir, f"{key}_history.json"))
                    if hist:
                        res["history"] = hist
                entry[key] = res
                print(f"  ✅ {key}: {result_dir}/{key}.json")
            else:
                print(f"  ⚠️  Non trovato: {result_dir}/{key}.json")

        if len(entry) == 1:  # solo "label", nessun dato
            print(f"  ❌ Run '{run}' saltato: nessun file trovato.\n")
            continue

        runs_data[run] = entry
        print()

    if not runs_data:
        print("❌ Nessun dato caricato. Controlla i path.")
        return

    print(f"📊 Generazione grafici in '{out}'...")
    plot_pretrain_total(runs_data, out)
    plot_pretrain_contrastive(runs_data, out)
    plot_pretrain_align(runs_data, out)
    plot_temperature(runs_data, out)
    plot_finetune_f1(runs_data, out)
    plot_finetune_acc(runs_data, out)
    plot_comparison_bar(runs_data, out)

    print(f"\n📋 Generazione CSV in '{out}'...")
    csv_pretrain_history(runs_data, out)
    csv_finetune_history(runs_data, out)
    csv_final_comparison(runs_data, out)

    print(f"\n✅ Tutto salvato in: {out}/")


if __name__ == "__main__":
    main()
