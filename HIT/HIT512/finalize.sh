#!/bin/bash
cd "$(dirname "$0")"
source ../../venv/bin/activate

while true; do
    done_workers=$(cat worker_gpu0.log worker_gpu1.log 2> /dev/null | grep -c "sem tarefas")
    python tables.py > results/tables.log 2>&1
    python plot.py > results/plot.log 2>&1
    echo "$(date '+%F %T') tabelas e figuras atualizadas" >> results/updates.log
    [ $done_workers -ge 2 ] && break
    sleep 7200
done
