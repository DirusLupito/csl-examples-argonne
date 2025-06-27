#!/usr/bin/env bash

set -e

cslc ./src/layout.csl --arch wse3 --fabric-dims=37,32 --fabric-offsets=4,1 \
--params=width:21,height:21,MAX_ZDIM:21 --params=BLOCK_SIZE:2 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py -m=21 -n=21 -k=21 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=21 --run-only --max-ite=400
