#!/usr/bin/env bash

set -e

cslc ./src/layout.csl --arch wse3 --fabric-dims=18,16 --fabric-offsets=4,1 \
--params=width:11,height:11,MAX_ZDIM:11 --params=BLOCK_SIZE:2 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py -m=11 -n=11 -k=11 --latestlink out --channels=1 --width-west-buf=0 --width-east-buf=0 --zDim=11 --run-only --max-ite=240
