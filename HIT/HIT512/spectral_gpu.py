"""Decaimento pseudo-espectral na GPU: spectral.py com hitgpu.

   Mesma integracao (RK3 de baixo armazenamento de Williamson, CFL 0.6, dt a cada
   5 passos) e mesmos tempos de saida (200 ao longo de 10 te), parando em 5 te.
   te e o da referencia (reference.txt = initial_0.txt), o mesmo em todas as realizacoes.
   Uso, dentro da pasta da realizacao: python ../../spectral_gpu.py <gpu>
"""

import os
import sys
import numpy as np
import cupy as cp
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hitgpu as H

cp.cuda.Device(int(sys.argv[1])).use()

t_end_te = 10
n_out    = 200
t_stop   = 5
cfl      = 0.6

P = H.read_params("initial.txt")
nu = float(P["nu"])
H.init(int(P["N"]), nu)
with h5py.File("initial.vtkhdf", "r") as f:
    u_hat = H.read_initial(f)
u_hat *= H.dealias
te0 = float(H.read_params("reference.txt")["te"])

A = (0.0, -5/9, -153/128)
B = (1/3, 15/16, 8/15)

t_out = np.linspace(0, t_end_te*te0, n_out)
t_out = t_out[t_out <= t_stop*te0 + 1e-9]

dt_visc = 1/(nu*3*H.kmax**2)
t, step, iout = 0.0, 0, 0
dt = min(H.compute_dt(u_hat, cfl), dt_visc)
residual = cp.zeros_like(u_hat)
rec = H.Recorder(te0)

while True:

    if t >= t_out[iout] - 1e-12:
        residual = None
        s = rec.add(t, u_hat)
        residual = cp.zeros_like(u_hat)
        print(f"out={iout:3d} t/te={t/te0:5.2f} ReL={s[5]:6.1f} ratio={np.mean(s[12+4*H.ndir:12+5*H.ndir]):.3f}", flush=True)
        iout += 1
        if iout == len(t_out):
            break

    if step % 5 == 0:
        dt = min(H.compute_dt(u_hat, cfl), dt_visc)
        if not np.isfinite(dt):
            sys.exit(f"NaN/inf em step={step} t={t:.6f}")

    dt_step = max(min(dt, t_out[iout] - t), 1e-12)

    for a, b in zip(A, B):
        r = H.nonlinear_hat(u_hat)
        for i in range(3):
            r[i] -= nu*H.k2*u_hat[i]
            residual[i] *= a
            residual[i] += dt_step*r[i]
            u_hat[i] += b*residual[i]
        del r

    t += dt_step
    step += 1

rec.save()
