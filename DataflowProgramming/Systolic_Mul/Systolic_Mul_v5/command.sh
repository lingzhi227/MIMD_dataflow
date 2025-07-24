#!/usr/bin/env bash

set -e

Pr=5
Pc=6
Cycle=4

Mt=5
Kt=5
Nt=5

M=$((Pr * Mt))
K=$((Cycle * Kt))
N=$((Pc * Nt))


cslc --arch=wse2 ./layout.csl --fabric-dims=11,7 \
--fabric-offsets=4,1 \
--params=M:${M},K:${K},N:${N},Mt:${Mt},Kt:${Kt},Nt:${Nt} \
--params=MEMCPYH2D_DATA_1_ID:0 \
--params=MEMCPYH2D_DATA_2_ID:1 \
--params=Pc:${Pc},Cycle:${Cycle},Pr:${Pr} \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 \
-o out


# trace instructions to sim.log file
# export SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace,landing
cs_python run.py --name out
