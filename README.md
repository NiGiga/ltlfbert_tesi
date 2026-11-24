# LTLfBERT

Contrastive pre-training of CodeBERT on LTLf (formula, trace) pairs.

## Struttura del Progetto

```text
LTLfBERT/
├── data/                      # Dataset JSON e split (train, val, test, ood)
│   ├── dataset/               # Dataset standard RandLTL (Spot)
│   ├── data_gp_island/        # Dataset generato con Genetic Programming (Island Model)
│   ├── data_gp/               # Dataset GP baseline
│   ├── dataset_5k/            # Dataset 5k
│   └── dataset_10k/           # Dataset 10k
├── src/                       # Codice sorgente Python
│   ├── data/                  # Generazione e validazione dataset
│   │   ├── generate_dataset.py
│   │   ├── evaluate_dataset.py
│   │   ├── test_is_empty.py
│   │   └── verify.py
│   ├── gp/                    # Algoritmi Genetici (DEAP) e confronto semantico
│   │   ├── gp_island.py       # Island Model Coarse-Grained (8 core, nicchie semantiche)
│   │   ├── gp_ltlf.py         # GP standard LTLf
│   │   ├── compare_datasets.py# Benchmark e confronto semantico
│   │   ├── generate_dataset.py# Generatore Spot RandLTL multithread
│   │   └── print_trees.py     # Ispezione visuale e alberi sintattici
│   ├── models/                # Architettura: dual-encoder + NT-Xent loss
│   ├── training/              # Pre-training contrastivo e fine-tuning
│   ├── evaluation/            # Valutazione: accuracy, F1, MRR, Recall@K, t-SNE
│   ├── utils/                 # Analisi teorica (Hopfield-Attention)
│   ├── run_experiment.py      # Pipeline end-to-end
│   ├── compare_orthogonality.py
│   └── analyze_results.py
├── slurm/                     # Job script ed esecuzione per Demetra HPC
│   ├── run_randltl.slurm      # Job Slurm per RandLTL 50k
│   ├── run_compare.slurm      # Job Slurm per confronto veloce
│   ├── run_hpc_pipeline.slurm # Pipeline completa su cluster Demetra
│   └── setup_demetra_env.sh   # Script setup ambiente Demetra (Conda/venv)
├── report.json                # Report metriche di confronto
├── report.md                  # Report esteso
└── requirements.txt
```

## Quick Start

```bash
# Test rapido su CPU
python src/run_experiment.py --quick --no-lora

# Run completo con LoRA su GPU
python src/run_experiment.py
```

## HPC Demetra (UniTS)

Gli script per lanciare i job su Slurm si trovano nella cartella `slurm/`:
```bash
# Esecuzione dalla root del repository:
sbatch slurm/run_randltl.slurm      # Generazione RandLTL 50k & Confronto
sbatch slurm/run_compare.slurm      # Solo confronto rapido
sbatch slurm/run_hpc_pipeline.slurm # Pipeline completa
```
