#!/bin/bash
#SBATCH --job-name=Systolic
#SBATCH --partition=cs
#SBATCH --account=app
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=python_job_%j.out
#SBATCH --mail-user=lingzhi.yang@stonybrook.edu
#SBATCH --mail-type=END,FAIL

# record start time
start_time=$(date +%s)

SIF=/shared/data1/Projects/Cerebras/cs_sdk_1.2.0/sdk-cbcore-202411041444-629-be7c2a56.sif

CS_PYTHON="singularity exec --bind $(realpath $PWD) --pwd $(realpath $PWD) $SIF python"

${CS_PYTHON} ./run.py --name out --cmaddr 172.24.140.100:9000

# cs_python ./run.py --name out --cmaddr 172.24.140.100:9000

# record end time
end_time=$(date +%s)

# compute runtime
runtime=$((end_time - start_time))

# format
runtime_formatted=$(printf '%02d:%02d:%02d\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60)))

echo "Program completed in: $runtime_formatted" >> python_job_${SLURM_JOB_ID}.out
