#!/bin/bash
#SBATCH --job-name=tensor_streaming
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:20:00
#SBATCH --output=python_job_%j.out

export PATH=/shared/data1/Projects/Cerebras/acceptance/hpc/cs_sdk_1.1.0:$PATH

cs_python ./run.py --name out --cmaddr 172.24.140.100:9000

