#!/bin/bash -l
#SBATCH --job-name=bootstrap_shapley_test
#SBATCH --output=bootstrap_shapley_%A_%a.log
#SBATCH --array=0-999%40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=02:30:00
#SBATCH --export=ALL

cd /home/fadoua.jouidel-amri/LES_paper_github || exit 1

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load libffi/3.4.4-GCCcore-13.2.0

source /home/fadoua.jouidel-amri/venvs/lesnb/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONWARNINGS="ignore::FutureWarning"

# sanity check in the batch shell
python -c "import ctypes; print('ctypes ok (batch)')"
echo "LD_LIBRARY_PATH (batch) = $LD_LIBRARY_PATH"

# run: force env export into the srun step
/usr/bin/time -v \
  srun --export=ALL /home/fadoua.jouidel-amri/venvs/lesnb/bin/python bootstrap_shp.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --data-dir /home/fadoua.jouidel-amri/LES_paper_github