#!/bin/bash
# ==============================================================================
# Slurm Job Script: Confronto Dataset (RandLTL vs GP Island)
# Cluster: Demetra HPC (UniTS)
# ==============================================================================
#
#SBATCH --job-name=ltlf_compare
#SBATCH --output=slurm/slurm_compare_%j.out
#SBATCH --error=slurm/slurm_compare_%j.err
#SBATCH --partition=Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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
echo "Avvio confronto dataset su $(hostname) at $(date)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-local}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-1}"
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
fi

echo "Using Python: $(which python3)"

# ------------------------------------------------------------------------------
# 2. Configurazione Percorsi
# ------------------------------------------------------------------------------
DIR_RANDLTL="data/dataset"
DIR_GP="data/data_gp_island"
REPORT_OUT="report.json"
MAX_LEN=100

echo ""
echo "======================================================================"
echo "Confronto in corso:"
echo "  • Dataset A (RandLTL):    $DIR_RANDLTL"
echo "  • Dataset B (GP Island):  $DIR_GP"
echo "  • Lunghezza temporale max: $MAX_LEN"
echo "  • File di report:         $REPORT_OUT"
echo "======================================================================"

python3 src/gp/compare_datasets.py \
    --a "$DIR_RANDLTL" \
    --b "$DIR_GP" \
    --max-len "$MAX_LEN" \
    --out "$REPORT_OUT"

echo ""
echo "======================================================================"
echo "Confronto completato con successo alle $(date)"
echo "Report salvato in: $REPORT_OUT"
echo "======================================================================"
