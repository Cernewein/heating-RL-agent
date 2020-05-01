#!/bin/sh
### General options
### --- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J heating-RL
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 30GB of system-memory
#BSUB -R "rusage[mem=30GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u cednewein@live.fr
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o training-%J.out
#BSUB -e training_%J.err
# -- end of LSF options --

# setup python3 env
module load python3/3.6.2
pip3 install --user virtualenv
virtualenv env
. env/bin/activate
python3 -m pip install --no-cache-dir -r requirement.txt

# running script
python3 main.py --model_name=$(date +%s | tail -c 8) --dynamic=True --noisy=False