#!/bin/bash
# ==============================================================================
# Slurm Job Script: LTLf Dataset Generation & Comparison Pipeline
# Cluster: Demetra HPC (UniTS)
# ==============================================================================
#
# Resources:
# - No GPU required: Dataset generation & comparison are CPU & memory bound.
# - CPU: 8 cores requested (for parallel matrix/array ops and Spot routines).
# - Memory: 32 GB RAM (sufficient for 50k formula matrices & trace pools).
# - Time: 24 hours (generates up to 50k formulas per generator and evaluates them).
#
#SBATCH --job-name=ltlf_gp_pipeline
#SBATCH --output=slurm/slurm_ltlf_%j.out
#SBATCH --error=slurm/slurm_ltlf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=Main
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nicola.gigante@studenti.units.it

set -e  # Exit immediately if a command fails
set -u  # Treat unset variables as an error
set -o pipefail

# Assicura che la working directory sia la root del repository LTLfBERT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "======================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-local}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-1}"
echo "Working directory: $(pwd)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Environment Activation
# ------------------------------------------------------------------------------
# Set OMP/MKL threads to match Slurm allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

# Option A: Conda / Micromamba environment (recommended for Spot)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate ltlfbert
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate ltlfbert
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ltlfbert
# Option B: Standard Python venv
elif [ -d "$HOME/envs/ltlfbert" ]; then
    source "$HOME/envs/ltlfbert/bin/activate"
elif [ -d "./.venv" ]; then
    source "./.venv/bin/activate"
else
    echo "ERROR: Could not find activated virtualenv or conda environment."
    echo "Please check the setup instructions in the README / guide."
    exit 1
fi

echo "Using Python: $(which python3)"
python3 -c "import spot, deap, numpy; print('Spot version:', spot.version() if hasattr(spot, 'version') else getattr(spot, '__version__', 'ok'), '| DEAP:', getattr(deap, '__version__', 'ok'), '| NumPy:', numpy.__version__)"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
N_FORMULAS=50000
MAX_LEN=100
DIR_RANDLTL="data/dataset"
DIR_GP="data/data_gp_island"
REPORT_OUT="report.json"

mkdir -p "$DIR_RANDLTL" "$DIR_GP"

# ------------------------------------------------------------------------------
# 2. Step 1: RandLTL Verified Dataset Generation
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "STEP 1: Generating RandLTL Dataset (${N_FORMULAS} formulas, max_len=${MAX_LEN})"
echo "Started at $(date)"
echo "======================================================================"

if [ -f "$DIR_RANDLTL/train.json" ]; then
    echo "Dataset RandLTL already exists in $DIR_RANDLTL. Skipping Step 1."
else
    python3 src/gp/generate_dataset.py \
        --n-formulas "$N_FORMULAS" \
        --traces-per-formula 4 \
        --max-len "$MAX_LEN" \
        --output-dir "$DIR_RANDLTL" \
        --n-workers "${SLURM_CPUS_PER_TASK:-8}" \
        --overwrite
fi

echo "Step 1 completed at $(date)"

# ------------------------------------------------------------------------------
# 3. Step 2: Genetic Programming Island Model (8 Cores Paralleli)
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "STEP 2: Generating GP Island Dataset (${N_FORMULAS} formulas, max_len=${MAX_LEN}, 8 Islands)"
echo "Started at $(date)"
echo "======================================================================"

python3 src/gp/gp_island.py \
    --n-formulas "$N_FORMULAS" \
    --max-len "$MAX_LEN" \
    --output-dir "$DIR_GP" \
    --n-islands 8 \
    --max-epochs 2000 \
    --resume

echo "Step 2 completed at $(date)"

# ------------------------------------------------------------------------------
# 4. Step 3: Dataset Comparison & Report
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "STEP 3: Comparing RandLTL vs GP Island Datasets (max_len=${MAX_LEN})"
echo "Started at $(date)"
echo "======================================================================"

python3 src/gp/compare_datasets.py \
    --a "$DIR_RANDLTL" \
    --b "$DIR_GP" \
    --max-len "$MAX_LEN" \
    --out "$REPORT_OUT"

echo "Step 3 completed at $(date)"

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "Pipeline execution finished successfully at $(date)"
echo "Outputs created:"
echo "  - RandLTL dataset: $DIR_RANDLTL/"
echo "  - GP dataset:      $DIR_GP/"
echo "  - Report:          $REPORT_OUT"
echo "======================================================================"
