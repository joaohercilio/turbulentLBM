#!/bin/bash
cd "$(dirname "$0")"

gpu=$1

tasks() {
    for r in $(seq 1 2); do echo "spectral/realization$r spectral $r"; done
    for ns in d3q15_3rd d3q19_3rd d3q27_3rd; do
        for b in 1 2 3; do for r in $(seq 1 2); do echo "lbm/$ns/Ma_x$b/realization$r lbm $r $ns $b"; done; done
    done
}

ready() {
    grep -qsaE "SNAPSHOT initial_$(($1 - 1))\.(h5|vtkhdf) " initial/chain*.log
}

while true; do
    left=0
    while read d kind r ns b; do
        [ -d $d ] && continue
        left=1
        ready $r || continue
        mkdir -p $(dirname $d)
        mkdir $d 2> /dev/null || continue
        echo "$(date '+%F %T') gpu$gpu inicio $d"
        if [ $kind = spectral ]; then ./run_spectral.sh $gpu $r; else ./run_lbm.sh $gpu $ns $b $r; fi
        echo "$(date '+%F %T') gpu$gpu fim    $d $(ls $d | grep -E '^(done|failed)$')"
        continue 2
    done < <(tasks)
    [ $left -eq 0 ] && break
    sleep 60
done
echo "$(date '+%F %T') gpu$gpu sem tarefas"
