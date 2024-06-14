#!/usr/bin/env bash
#SBATCH --job-name=phyFea_segformer_training
#SBATCH --output=/cluster/work/cvl/shbasu/phyfeaSegformer/results/phyFea_segformer_training_output.log
#SBATCH --error=/cluster/work/cvl/shbasu/phyfeaSegformer/results/phyFea_segformer_training_error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shamik.basu@studio.unibo.it
#SBATCH --gpus=8
#SBATCH --gres=gpumem:40g
#SBATCH --mem-per-cpu=8G
#sbatch --tmp=80G
#SBATCH --ntasks=8

CONFIG=$1
GPUS=8
PORT=15661



CUDA_HOME=/cluster/apps/gcc-6.3.0/cuda-11.0.3-qdlibd2luz2fy7izfefao4c5yitxwjus

source /cluster/apps/local/env2lmod.sh
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy

/var/spool/slurm/slurmd/state/job62036655/


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname $0)/SegFormer/tools/train.py\
   $(dirname $0)/SegFormer/local_configs/segformer/B4/phyFeaSegformer_B4_city_1024.py\
 --work-dir=/cluster/work/cvl/shbasu/phyfeaSegformer/results\
 --launcher='pytorch' --load-from='/cluster/work/cvl/shbasu/phyfeaSegformer/models/pretrained_models/mit_b4.pth'\
 --resume-from='/cluster/work/cvl/shbasu/phyfeaSegformer/models/pretrained_models/mit_b4.pth'\
 --gpus=8 --seed=2022