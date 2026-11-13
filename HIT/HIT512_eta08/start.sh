#!/bin/bash
cd "$(dirname "$0")"
source ../../venv/bin/activate
if [ $1 -eq 0 ]; then
    (cd initial && python ../initial_gpu.py 0 ../../HIT512/initial/initial_0.vtkhdf 0 2 > chainA.log 2>&1)
else
    (cd initial && python ../initial_gpu.py 1 ../../HIT512/initial/initial_4.vtkhdf 2 2 > chainB.log 2>&1)
fi
./worker.sh $1 >> worker_gpu$1.log 2>&1
