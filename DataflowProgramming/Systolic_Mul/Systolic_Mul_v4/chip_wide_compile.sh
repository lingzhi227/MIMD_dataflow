#!/usr/bin/env bash

# export PATH=/shared/data1/Projects/Cerebras/acceptance/hpc/cs_sdk_1.1.0:$PATH
# export PATH=/shared/data1/Projects/Cerebras/acceptance/hpc/cs_sdk_1.2.0:$PATH
export PATH=/shared/data1/Projects/Cerebras/cs_sdk_1.2.0:$PATH

set -e

Pr=993
Pc=749
Cycle=700

Mt=74
Kt=74
Nt=74

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
