#!/usr/bin/env bash

set -e

# M=20
# K=20
# N=20
# Mt=20
# Kt=20
# Nt=20


# Pr=$((M / Mt))
# Cycle=$((K / Kt))
# Pc=$((N / Nt))

Pr=1
Pc=2
Cycle=5

Mt=74
Kt=74
Nt=74

M=$((Pr * Mt))
K=$((Cycle * Kt))
N=$((Pc * Nt))



cslc --arch=wse2 ./layout.csl --fabric-dims=30,30 \
--fabric-offsets=4,1 \
--params=M:${M},K:${K},N:${N},Mt:${Mt},Kt:${Kt},Nt:${Nt} \
--params=MEMCPYH2D_DATA_1_ID:0 \
--params=MEMCPYH2D_DATA_2_ID:1 \
--params=Pc:${Pc},Cycle:${Cycle},Pr:${Pr} \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 \
-o out

export SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace@P5.2,landing@P5.2
cs_python run.py --name out
