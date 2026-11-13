"""Spin-up forcado na GPU continuando de uma IC colhida: initial.py com hitgpu.

   Mesmo forcamento de Jimenez, Wray, Saffman & Rogallo (JFM 255, 1993), mesma lei
   nu_f = eps_alvo/(2*Om_forcado) com eps_alvo = nu^3/eta_alvo^4, reajuste a cada
   5 passos. A fonte ja esta nas unidades reescaladas (a congelado na 1a colheita):
   pela simetria u -> a*u, nu -> a*nu, t -> t/a a dinamica forcada e a mesma, e
   aqui a = 1. te0 = t(initial_0)/t_spinup, e colhe a cada t_space*te0.
   perturbacao > 0 soma ruido solenoidal de amplitude relativa perturbacao (ramo
   independente: o caos descorrelaciona em poucos te).
   Uso: python initial_gpu.py <gpu> <fonte.vtkhdf> <primeiro_indice> <n_snap> [perturbacao]
"""

import os
import sys
import numpy as np
import cupy as cp
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hitgpu as H

cp.cuda.Device(int(sys.argv[1])).use()
source, first, n_snap = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
perturbation = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

eta_delta_target = 2.0
k_force  = 2.5
t_spinup = 15
t_space  = 10
cfl      = 0.3

P = H.read_params(H.params_path(source))
nu, t0, N = float(P["nu"]), float(P["t"]), int(P["N"])
keep = {key: P[key] for key in ("seed", "k_force", "rescale")}
with h5py.File(source, "r") as f:
    nu *= (eta_delta_target/1.1)**(4/3)
    H.init(N, nu)
    u_hat = H.read_initial(f)
u_hat *= H.dealias
te0 = float(H.read_params("../../HIT512/initial/initial_0.txt")["t"])/t_spinup

if perturbation > 0:
    urms = H.diagnostics(u_hat)["urms"]
    noise = cp.random.default_rng(first).standard_normal((3, N, N, N))*perturbation*urms
    u_hat += cp.stack([H.fwd(noise[i]) for i in range(3)])
    del noise
    pk = (H.k[0]*u_hat[0] + H.k[1]*u_hat[1] + H.k[2]*u_hat[2])*H.inv_k2
    for i in range(3):
        u_hat[i] -= H.k[i]*pk
    u_hat *= H.dealias
    del pk

eta = eta_delta_target*H.delta
eps_target = nu**3/eta**4
k2_forced = cp.where((H.k2 > 0) & (H.k2 < k_force**2), H.k2, 0.0)
nu_f = 0.0

def update_forcing():
    global nu_f
    e = 0.5*cp.sum(cp.abs(u_hat)**2, axis=0)*H.w
    Om_forced = float(cp.sum(e*k2_forced))
    nu_f = eps_target/(2*Om_forced) if Om_forced > 0 else 0.0

A = (0.0, -5/9, -153/128)
B = (1/3, 15/16, 8/15)

dt_visc = 1/(nu*3*H.kmax**2)
t, step, isnap = 0.0, 0, 0
t_save = [(i + 1)*t_space*te0 for i in range(n_snap)]
residual = cp.zeros_like(u_hat)
d = H.diagnostics(u_hat)
print(f"fonte={source} t0={t0:.4f} te0={te0:.4f} nu={nu:.6e} eps_alvo={eps_target:.4e} "
      f"perturbacao={perturbation:g} colheitas em t/te0 = {[round((t0 + x)/te0, 2) for x in t_save]}", flush=True)

while True:

    if t >= t_save[isnap] - 1e-12:
        residual = None
        d, lbm = H.harvest(f"initial_{first + isnap}.vtkhdf", u_hat,
                           dict(keep, t=t0 + t, source=source, perturbation=perturbation))
        print(f"SNAPSHOT initial_{first + isnap}.vtkhdf t/te0={(t0 + t)/te0:.2f} ReL={d['ReL']:.1f} "
              f"eta/delta={d['eta_delta']:.3f} urms={d['urms']:.6f} te={d['te']:.2f} tau={lbm['tau']:.6f}", flush=True)
        isnap += 1
        if isnap == n_snap:
            break
        residual = cp.zeros_like(u_hat)

    if step % 5 == 0:
        dt = min(H.compute_dt(u_hat, cfl), dt_visc)
        if not np.isfinite(dt):
            sys.exit(f"NaN/inf em step={step} t={t:.6f}")
        pk = (H.k[0]*u_hat[0] + H.k[1]*u_hat[1] + H.k[2]*u_hat[2])*H.inv_k2
        for i in range(3):
            u_hat[i] -= H.k[i]*pk
        del pk
        update_forcing()
        d = H.diagnostics(u_hat)

    dt_step = max(min(dt, t_save[isnap] - t), 1e-12)

    for a, b in zip(A, B):
        r = H.nonlinear_hat(u_hat)
        for i in range(3):
            r[i] += (nu_f*k2_forced - nu*H.k2)*u_hat[i]
            residual[i] *= a
            residual[i] += dt_step*r[i]
            u_hat[i] += b*residual[i]
        del r

    t += dt_step
    step += 1

    if step % 100 == 0:
        print(f"step={step} t/te0={(t0 + t)/te0:.2f} ReL={d['ReL']:.1f} eta/delta={d['eta_delta']:.3f} "
              f"urms={d['urms']:.6f} L11={d['L11']:.3f} nu_f/nu={nu_f/nu:.2f} dt={dt:.3e}", flush=True)
