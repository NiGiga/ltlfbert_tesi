#!/usr/bin/env python3
"""
LTLfBERT - Script principale dell'esperimento completo.
Esegue in sequenza:
  1. Generazione dataset
  2. Pre-training contrastivo LTLfBERT
  3. Fine-tuning: baseline vs LTLfBERT
  4. Valutazione e confronto
"""

import argparse
import json
import os
import torch, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    p = argparse.ArgumentParser(description="LTLfBERT Experiment Runner")
    p.add_argument("--quick", action="store_true", help="Modalità rapida per CPU/Debug")
    p.add_argument("--no-lora", action="store_true", help="Disabilita LoRA")
    p.add_argument("--force-data", action="store_true", help="Forza la rigenerazione del dataset")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size per il training")
    p.add_argument("--formulas", type=int, default=20000, help="Numero formule (solo se --force-data)")
    p.add_argument("--skip-pretrain", action="store_true", help="Salta il pre-training")
    p.add_argument("--skip-baseline", action="store_true", help="Salta il fine-tuning baseline")
    args = p.parse_args()

    # ── Parametri e Cartelle Dinamiche ──────────────────────────────────────
    base_dir = "ltlfbert"

    # Setup ambienti
    mode = "quick" if args.quick else "comp"
    DATA = os.path.join(base_dir, f"data_{mode}")
    CKPT = os.path.join(base_dir, f"checkpoints_{mode}")
    OUTDIR = os.path.join(base_dir, f"results_{mode}")

    # Configurazione Iperparametri
    N_FORMULAS = 5000 if args.quick else args.formulas
    PRETRAIN_EPOCHS = 5 if args.quick else 30
    FINETUNE_EPOCHS = 3 if args.quick else 15
    BATCH = 16 if args.quick else args.batch_size

    print(f"🚀 AVVIO RUN {mode.upper()}: Salvo in {OUTDIR} | Batch: {BATCH}")

    os.makedirs(DATA, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)

    USE_LORA = not args.no_lora

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\nSTEP 1: Generazione Dataset\n" + "=" * 60)
    train_file = os.path.join(DATA, "train.json")
    if os.path.exists(train_file) and not args.force_data:
        print(f"✅ Dataset trovato in '{DATA}'. Generazione saltata.")
    else:
        print(f"⏳ Generazione di {N_FORMULAS} formule...")
        from data.generate_dataset import generate_dataset
        generate_dataset(n_formulas=N_FORMULAS, output_dir=DATA)

    # ── 2. Pre-training ──────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\nSTEP 2: Pre-training Contrastivo (LTLfBERT)\n" + "=" * 60)
    if not args.skip_pretrain:
        from training.pretrain import pretrain
        pretrain(
            data_dir=DATA, output_dir=CKPT,
            epochs=PRETRAIN_EPOCHS, batch_size=BATCH, use_lora=USE_LORA,
        )
    else:
        print("⏭️  Pre-training saltato.")



    # ── Pulizia VRAM tra pretraining e fine-tuning ──

    gc.collect()
    torch.cuda.empty_cache()
    print(f"VRAM libera prima del fine-tuning: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

    # ── 3. Fine-tuning ────────────────────────────────────────────────────────
    from training.finetune import finetune_conformance

    if not args.skip_baseline:
        print("\n" + "=" * 60 + "\nSTEP 3a: Fine-tuning Baseline (CodeBERT puro)\n" + "=" * 60)
        res_base = finetune_conformance(
            data_dir=DATA, checkpoint_path=None, output_dir=OUTDIR,
            run_name="baseline", epochs=FINETUNE_EPOCHS, batch_size=BATCH,
            use_lora=False, lr=1e-4,
        )
    else:
        base_path = os.path.join(OUTDIR, "baseline_results.json")
        res_base = json.load(open(base_path)) if os.path.exists(base_path) else {"test_acc": 0, "test_f1": 0,
                                                                                 "ood_acc": 0, "ood_f1": 0}
        print("⏭️  Baseline saltata, carico risultati precedenti.")

    # STEP 3b — sempre eseguito, indipendente da --skip-baseline
    print("\n" + "=" * 60 + "\nSTEP 3b: Fine-tuning LTLfBERT pre-addestrato\n" + "=" * 60)
    res_ltlf = finetune_conformance(
        data_dir=DATA, checkpoint_path=os.path.join(CKPT, "ltlfbert_best.pt"),
        output_dir=OUTDIR, run_name="ltlfbert", epochs=FINETUNE_EPOCHS,
        batch_size=max(8, BATCH // 2), use_lora=USE_LORA, lr=3e-5,
    )



    # ── 4. Valutazione & Confronto ────────────────────────────────────────────
    print("\n" + "=" * 60 + "\nSTEP 4: Valutazione Completa\n" + "=" * 60)
    from evaluation.evaluate import full_evaluation
    full_evaluation(
        data_dir=DATA,
        baseline_ckpt=os.path.join(OUTDIR, "baseline_best.pt"),
        ltlfbert_ckpt=os.path.join(OUTDIR, "ltlfbert_best.pt"),
        output_dir=OUTDIR,
    )

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    if 'res_ltlf' in locals() and 'res_base' in locals():
        delta_acc = (res_ltlf['test_acc'] - res_base['test_acc']) * 100
        delta_f1 = (res_ltlf['test_f1'] - res_base['test_f1']) * 100
        print("\n" + "=" * 60 + "\nRIEPILOGO FINALE\n" + "=" * 60)
        print(f"  Δ accuracy = {delta_acc:+.2f}pp | Δ F1 = {delta_f1:+.2f}pp")
        print(f"  OOD Generalizzazione: Baseline {res_base['ood_acc']:.4f} → LTLfBERT {res_ltlf['ood_acc']:.4f}")
        summary = {"baseline": res_base, "ltlfbert": res_ltlf, "delta_acc_pp": delta_acc}
        with open(os.path.join(OUTDIR, "final_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nRisultati salvati in: {OUTDIR}/")
    else:
        print("⚠️  Riepilogo saltato: risultati non disponibili.")


if __name__ == "__main__":
    main()