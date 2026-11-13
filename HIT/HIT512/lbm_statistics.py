"""Estatisticas dos snapshots do LBM (fields.h5) com hitgpu.statistics.

   Com o atributo scale = b (Mach b vezes o da IC) volta as unidades do espectral:
   u -> u/b e t = b*passo*delta (simetria u -> b*u, nu -> b*nu, t -> t/b).

   Uso, dentro da pasta da realizacao: python ../../lbm_statistics.py <gpu>
"""

import os
import sys
import numpy as np
import cupy as cp
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hitgpu as H

cp.cuda.Device(int(sys.argv[1])).use()

P = H.read_params("initial.txt")
H.init(int(P["N"]), float(P["nu"]))
te0 = float(H.read_params("reference.txt")["te"])

steps = np.loadtxt("times.dat", ndmin=2)[:, 1]
rec = H.Recorder(te0)

with h5py.File("fields.h5", "r") as f:
    b = float(f.attrs.get("scale", 1.0))
    for iout, step in enumerate(steps):
        u_hat = H.read_velocity(f[f"u/3D/{iout}"])/b
        s = rec.add(b*step*H.delta, u_hat)
        print(f"out={iout:3d} t/te={b*step*H.delta/te0:5.2f} ReL={s[5]:6.1f} "
              f"ratio={np.mean(s[12+4*H.ndir:12+5*H.ndir]):.3f}", flush=True)

rec.save()
