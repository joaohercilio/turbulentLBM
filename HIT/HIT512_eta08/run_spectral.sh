#!/bin/bash
cd "$(dirname "$0")"
source ../../venv/bin/activate

gpu=$1
r=$2
d=spectral/realization$r
ln -sfn ../../initial/initial_$((r - 1)).vtkhdf $d/initial.vtkhdf
ln -sfn ../../initial/initial_$((r - 1)).txt $d/initial.txt
ln -sfn ../../initial/initial_0.txt $d/reference.txt
(cd $d && python ../../spectral_gpu.py $gpu > log.txt 2>&1) && touch $d/done || touch $d/failed
