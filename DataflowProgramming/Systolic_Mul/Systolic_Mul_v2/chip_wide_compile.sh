#!/usr/bin/env bash

# export PATH=/shared/data1/Projects/Cerebras/acceptance/hpc/cs_sdk_1.1.0:$PATH
# export PATH=/shared/data1/Projects/Cerebras/acceptance/hpc/cs_sdk_1.2.0:$PATH
export PATH=/shared/data1/Projects/Cerebras/cs_sdk_1.2.0:$PATH

set -e

# M=5000
# N=500
# K=5000
# Mt=10
# Kt=10
# Nt=10

# Pr=$((M / Mt))
# Cycle=$((K / Kt))
# Pc=$((N / Nt))

Pr=5
Pc=5
Cycle=5

Mt=5
Kt=5
Nt=5

M=$((Pr * Mt))
K=$((Cycle * Kt))
N=$((Pc * Nt))


cslc --arch=wse2 ./layout.csl --fabric-dims=757,996 \
--fabric-offsets=4,1 \
--params=M:${M},K:${K},N:${N},Mt:${Mt},Kt:${Kt},Nt:${Nt} \
--params=MEMCPYH2D_DATA_1_ID:0 \
--params=MEMCPYH2D_DATA_2_ID:1 \
--params=Pc:${Pc},Cycle:${Cycle},Pr:${Pr} \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 \
-o out
