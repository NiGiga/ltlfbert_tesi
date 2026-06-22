# LTLfBERT

Contrastive pre-training of CodeBERT on LTLf (formula, trace) pairs.

## Struttura
```
ltlfbert/
├── data/generate_dataset.py   # Generatore dataset con valutatore LTLf puro-Python
├── models/ltlfbert.py         # Architettura: dual-encoder + NT-Xent loss
├── training/pretrain.py       # Pre-training contrastivo
├── training/finetune.py       # Fine-tuning Task B (conformance detection)
├── evaluation/evaluate.py     # Metriche: accuracy, F1, MRR, Recall@K, t-SNE
├── utils/hopfield_analysis.py # Collegamento teorico Hopfield–Attention
└── run_experiment.py          # Script principale (end-to-end)
```

## Quick Start

```bash
# Test rapido su CPU (pochi minuti)
cd ~/ltlfbert
python run_experiment.py --quick --no-lora

# Run completo con LoRA su GPU
python run_experiment.py
```

## Dipendenze
```
torch transformers peft scikit-learn matplotlib
```

## Pipeline
1. `generate_dataset.py` → genera ~10K coppie (formula LTLf, traccia)
2. `pretrain.py`         → pre-training NT-Xent su coppie positive
3. `finetune.py`         → fine-tuning baseline vs LTLfBERT su Task B
4. `evaluate.py`         → MRR, Recall@K, t-SNE del latent space
