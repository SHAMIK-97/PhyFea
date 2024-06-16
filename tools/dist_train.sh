#!/usr/bin/env bash
#SBATCH --job-name=phyFea_segformer_training
#SBATCH --output=/cluster/work/cvl/shbasu/phyfeaSegformer/results/phyFea_segformer_training_output.log
#SBATCH --error=/cluster/work/cvl/shbasu/phyfeaSegformer/results/phyFea_segformer_training_error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shamik.basu@studio.unibo.it
#SBATCH --gpus=rtx_3090:4
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks=4
#SBATCH --time=1:00:00

CONFIG=$1
GPUS=4
PORT=$(shuf -i 15661-55661 -n 1)

source /cluster/apps/local/env2lmod.sh
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy





PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    /cluster/work/cvl/shbasu/phyfeaSegformer/models/SegFormer/tools/train.py\
   /cluster/work/cvl/shbasu/phyfeaSegformer/models/SegFormer/local_configs/segformer/B4/phyFeaSegformer_B4_city_1024.py\
 --work-dir=/cluster/work/cvl/shbasu/phyfeaSegformer/results\
 --launcher='pytorch' --load-from='/cluster/work/cvl/shbasu/phyfeaSegformer/models/pretrained_models/mit_b4.pth'\
 --gpus=4 --seed=2022
