#!/usr/bin/env bash

set -e

M=94000
K=74000
N=74000
Mt=80
Kt=80
Nt=80


Pr=$((M / Mt))
Cycle=$((K / Kt))
Pc=$((N / Nt))


cslc --arch=wse2 ./layout.csl --fabric-dims=757,996 \
--fabric-offsets=4,1 \
--params=M:${M},K:${K},N:${N},Mt:${Mt},Kt:${Kt},Nt:${Nt} \
--params=MEMCPYH2D_DATA_1_ID:0 \
--params=MEMCPYH2D_DATA_2_ID:1 \
--params=Pc:${Pc},Cycle:${Cycle},Pr:${Pr} \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 \
-o out

