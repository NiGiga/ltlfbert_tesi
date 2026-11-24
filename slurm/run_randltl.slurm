#!/bin/bash
# ==============================================================================
# Slurm Job Script: Generazione RandLTL (50k formule) & Confronto con GP Island
# Cluster: Demetra HPC (UniTS)
# ==============================================================================
#
#SBATCH --job-name=ltlf_randltl
#SBATCH --output=slurm/slurm_randltl_%j.out
#SBATCH --error=slurm/slurm_randltl_%j.err
#SBATCH --partition=Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nicola.gigante@studenti.units.it

set -e
set -u
set -o pipefail

# Assicura che la working directory sia la root del repository LTLfBERT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "======================================================================"
echo "Avvio generazione RandLTL su $(hostname) at $(date)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-local}"
echo "CPUs allocati: ${SLURM_CPUS_PER_TASK:-8}"
echo "Working directory: $(pwd)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Attivazione Ambiente Conda
# ------------------------------------------------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate ltlfbert
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate ltlfbert
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ltlfbert
elif [ -d "$HOME/envs/ltlfbert" ]; then
    source "$HOME/envs/ltlfbert/bin/activate"
fi

echo "Using Python: $(which python3)"

# ------------------------------------------------------------------------------
# 2. Generazione RandLTL (50.000 Formule, 4 tracce per formula, max_len=100)
# ------------------------------------------------------------------------------
N_FORMULAS=50000
MAX_LEN=100
TRACES_PER_FORMULA=4
DIR_RANDLTL="data/dataset"
DIR_GP="data/data_gp_island"
REPORT_OUT="report.json"

echo ""
echo "======================================================================"
echo "STEP 1: Generazione RandLTL in corso..."
echo "  • Formule richieste:     $N_FORMULAS"
echo "  • Tracce per formula:    $TRACES_PER_FORMULA (bilanciate 50/50 pos/neg)"
echo "  • Orizzonte temporale:   max_len=$MAX_LEN"
echo "  • Cartella output:       $DIR_RANDLTL"
echo "  • Processi paralleli:    ${SLURM_CPUS_PER_TASK:-8}"
echo "======================================================================"

python3 src/gp/generate_dataset.py \
    --n-formulas "$N_FORMULAS" \
    --traces-per-formula "$TRACES_PER_FORMULA" \
    --max-len "$MAX_LEN" \
    --output-dir "$DIR_RANDLTL" \
    --n-workers "${SLURM_CPUS_PER_TASK:-8}" \
    --overwrite

echo ""
echo "Step 1 completato con successo alle $(date)"

# ------------------------------------------------------------------------------
# 3. Confronto Automatico RandLTL (50k) vs GP Island (50k)
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "STEP 2: Confronto RandLTL vs GP Island..."
echo "  • Dataset A: $DIR_RANDLTL"
echo "  • Dataset B: $DIR_GP"
echo "  • Report:    $REPORT_OUT"
echo "======================================================================"

python3 src/gp/compare_datasets.py \
    --a "$DIR_RANDLTL" \
    --b "$DIR_GP" \
    --max-len "$MAX_LEN" \
    --out "$REPORT_OUT"

echo ""
echo "======================================================================"
echo "Tutto completato con successo alle $(date)!"
echo "Report finale salvato in: $REPORT_OUT"
echo "======================================================================"
