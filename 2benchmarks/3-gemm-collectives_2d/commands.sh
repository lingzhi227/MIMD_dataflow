#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./layout.csl --fabric-dims=757,996 --fabric-offsets=4,1 \
--params=P:750,Mt:40,Kt:40,Nt:40 \
--memcpy --channels=1 -o out
cs_python run.py --name out
