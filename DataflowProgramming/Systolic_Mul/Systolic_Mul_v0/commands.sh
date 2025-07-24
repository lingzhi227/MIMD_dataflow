#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=M:15,K:12,N:16,Mt:3,Kt:3,Nt:4 \
--memcpy --channels=1 -o out
cs_python run.py --name out
