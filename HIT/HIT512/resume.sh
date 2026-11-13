#!/bin/bash
cd "$(dirname "$0")"
source ../../venv/bin/activate
if [ $1 -eq 0 ]; then
    (cd initial && python ../initial_gpu.py 0 initial_2.vtkhdf 3 2 > chainA2.log 2>&1; touch chainA.done)
else
    (cd initial && python ../initial_gpu.py 1 initial_6.vtkhdf 7 1 > chainB2.log 2>&1; touch chainB.done)
fi
./worker.sh $1 >> worker_gpu$1.log 2>&1
