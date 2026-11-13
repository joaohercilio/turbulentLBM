#!/usr/bin/env bash
# Video final: onda real em cima (em loop, sem legenda), simulacao embaixo, DUR segundos.
# Uso: ./compose.sh [video real]   (variaveis: DUR, START, END, SIM)
set -euo pipefail
cd "$(dirname "$0")"
REAL=${1:-$HOME/Downloads/rivs.mp4}
SIM=${SIM:-frames/sim}
SUB=4                        # quadros renderizados por saida (render.py)
START=${START:-6}            # primeira saida (antes disso Q < 0 em todo lugar)
END=${END:-100}              # ultima saida (t = 25)
DUR=${DUR:-15}
FPS=30
W=1280; H=720
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
N=$(( (END - START)*SUB ))
RATE=$(python3 -c "print($N/$DUR)")
TITLE='Temporal mixing layer (Moment Representation LBM, D3Q27 3rd-order regularization)'

ffmpeg -y -v error \
  -stream_loop -1 -i "$REAL" \
  -framerate "$RATE" -start_number $((START*SUB)) -i "$SIM/f%05d.png" \
  -filter_complex "\
[0:v]scale=$W:$H:force_original_aspect_ratio=increase,crop=$W:$H,fps=$FPS,setpts=PTS-STARTPTS[r];\
[1:v]scale=$W:$H,fps=$FPS,setpts=PTS-STARTPTS,\
drawtext=fontfile=$FONT:text='$TITLE':x=24:y=20:fontsize=25:fontcolor=white,\
drawtext=fontfile=$FONT:text='Q isosurfaces':x=24:y=56:fontsize=20:fontcolor=white@0.8,\
drawtext=fontfile=$FONT:text='t U/δω = %{eif\:($START + n*$N/($DUR*$FPS)/$SUB)*0.25\:d}.%{eif\:mod(($START*$SUB + n*$N/($DUR*$FPS))*25/$SUB\,100)/10\:d}':\
x=w-tw-22:y=h-th-20:fontsize=26:fontcolor=white:box=1:boxcolor=0x02080f:boxborderw=10[s];\
[r][s]vstack,fade=t=in:st=0:d=0.8,fade=t=out:st=$(python3 -c "print($DUR-0.8)"):d=0.8[v]" \
  -map "[v]" -t "$DUR" -an -c:v libx264 -pix_fmt yuv420p -crf 17 -r $FPS ribs_lbm_vs_wave.mp4
echo "ok: $(pwd)/ribs_lbm_vs_wave.mp4"
