#!/bin/bash

python main.py --arch wrn --depth 16 --widening_factor 4 --drop_rate 0.3 --epochs 50 --scheduler cosine --use_random_erasing --test_style microsoft --outdir ./results/wrnmicro
