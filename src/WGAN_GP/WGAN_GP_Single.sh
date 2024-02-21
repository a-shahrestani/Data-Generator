#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100l:1 # request a GPU
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=4 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=32G      
#SBATCH --time=24:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-gargoum
#SBATCH --mail-user=afshin.shahrestani@ubc.ca
#SBATCH --mail-type=ALL

module load python/3.10 # Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision numpy pandas tqdm matplotlib --no-index
cp -R /project/def-gargoum/afshinsh/datasets/Pavement\ Crack\ Detection/Crack500-Forest\ Annotated $SLURM_TMPDIR/Data

echo "starting training..."
time python WGAN_GP_Single.py --batch_size=32 --num_workers=3