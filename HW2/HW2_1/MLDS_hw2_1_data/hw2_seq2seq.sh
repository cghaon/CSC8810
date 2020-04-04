#!/bin/sh

python s2vt.py ("$1")
python MLDS_hw2_1_data/bleu_eval.py ("$2")
