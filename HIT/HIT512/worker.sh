#!/bin/bash
cd "$(dirname "$0")"
source ../../venv/bin/activate
H=$PWD
gpu=$1

tasks() {
    for r in $(seq 8); do echo spectral/realization$r $r; done
    for b in 1 2 3; do
        for ns in d3q15_2nd d3q19_2nd d3q27_2nd d3q15_3rd d3q19_3rd d3q27_3rd; do
            for r in $(seq 8); do echo lbm/$ns/Ma_x$b/realization$r $r $ns $b; done
        done
    done
}

run() {
    d=$1; r=$2; ns=$3; b=$4
    ln -sfn $H/initial/initial_$((r - 1)).vtkhdf $d/initial.vtkhdf
    cd $d
    if [ -z "$ns" ]; then
        python $H/pseudo_spectral_gpu.py $gpu > log.txt 2>&1
    else
        $H/../../build/${ns}_x${b}_gpu$gpu/hip_test > log.txt 2>&1 && python $H/lbm_statistics.py $gpu > statistics.log 2>&1
    fi && touch done || touch failed
    rm -f fields.h5
}

while true; do
    left=0
    while read d r ns b; do
        [ -d $d ] && continue
        left=1
        [ -f initial/initial_$((r - 1)).vtkhdf ] || continue
        mkdir -p $(dirname $d)
        mkdir $d 2> /dev/null || continue
        echo "$(date '+%F %T') gpu$gpu inicio $d"
        (run $d $r $ns $b) < /dev/null
        echo "$(date '+%F %T') gpu$gpu fim    $d $(ls $d | grep -xE 'done|failed')"
        continue 2
    done < <(tasks)
    [ $left -eq 0 ] && break
    sleep 60
done
echo "$(date '+%F %T') gpu$gpu sem tarefas"
