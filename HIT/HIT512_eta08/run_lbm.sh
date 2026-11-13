#!/bin/bash
cd "$(dirname "$0")"
source ../../venv/bin/activate

gpu=$1
ns=$2
b=$3
r=$4
d=lbm/$ns/Ma_x$b/realization$r
ln -sfn ../../../../initial/initial_$((r - 1)).vtkhdf $d/initial.vtkhdf
ln -sfn ../../../../initial/initial_$((r - 1)).txt $d/initial.txt
ln -sfn ../../../../initial/initial_0.txt $d/reference.txt
(cd $d && ../../../../../../build/${ns}_x${b}_gpu$gpu/hip_test > log.txt 2>&1 \
    && python ../../../../lbm_statistics.py $gpu > statistics.log 2>&1) && touch $d/done || touch $d/failed
rm -f $d/fields.h5
