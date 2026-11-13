#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "^python /home/geoenergialab/turbulentLBM/HIT/HIT512/spectral_gpu.py" > /dev/null; do sleep 30; done
source ../../venv/bin/activate
(cd initial && python ../initial_gpu.py 1 initial_0.vtkhdf 5 3 1e-2 > chainB.log 2>&1; touch chainB.done)
./worker.sh 1
