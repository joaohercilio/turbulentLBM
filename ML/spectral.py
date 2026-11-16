"""Solucao de referencia pseudo-espectral do caso de problems/shearLayer.hpp, na GPU (CuPy).

   Le o mesmo initial.vtkhdf (PointData com shape [NZ, NY, NX]) do LBM; parametros iguais aos de ML/initial.py. Deslizamento livre nas paredes equivale a espelhar o dominio em y
   (u, w pares e v impar): as linhas de fluido j = 1..NY-2 e a sua imagem formam um dominio periodico
   de altura 2(NY - 2), em que os pontos coincidem com os nos do LBM. Unidades de rede (dx = 1).
   Mesma integracao do HIT: RK3 de baixo armazenamento de Williamson, forma rotacional, regra dos 2/3,
   dt a cada 5 passos. Grava fields_XXXX.vtkhdf a cada delta_w/U, no mesmo formato e na mesma malha
   (incluindo as linhas fantasma) do LBM.
   Uso, na pasta com initial.vtkhdf: python spectral.py <gpu>
"""

import sys
import numpy as np
import cupy as cp
import h5py

cp.cuda.Device(int(sys.argv[1])).use()

t_end, cfl = 20, 0.5

NX, NY, NZ = 256, 768, 154
U, RE = 0.05, 500
DW = NX/(1.16*2*np.pi)
nu = U*DW/RE
T_REF = DW/U

with h5py.File("initial.vtkhdf", "r") as f:
    u0 = [np.ascontiguousarray(f["VTKHDF/PointData/" + k][:, 1:NY - 1, :].transpose(2, 1, 0)) for k in ("ux", "uy", "uz")]

H = NY - 2
shape = (NX, 2*H, NZ)
fwd = lambda a: cp.fft.rfftn(a, norm="forward")
bwd = lambda a: cp.fft.irfftn(a, s=shape, norm="forward")

n = [cp.fft.fftfreq(NX, 1/NX), cp.fft.fftfreq(2*H, 1/(2*H)), cp.fft.rfftfreq(NZ, 1/NZ)]
k = [2*cp.pi*n[0][:, None, None]/NX, 2*cp.pi*n[1][None, :, None]/(2*H), 2*cp.pi*n[2][None, None, :]/NZ]
k2 = k[0]**2 + k[1]**2 + k[2]**2
inv_k2 = cp.where(k2 != 0, 1/cp.where(k2 != 0, k2, 1), 0)
dealias = ((cp.abs(n[0]) < NX//3)[:, None, None] & (cp.abs(n[1]) < (2*H)//3)[None, :, None]
           & (cp.abs(n[2]) < NZ//3)[None, None, :])
kmax = float(cp.sqrt(cp.max(k2*dealias)))


def project(a):
    p = (k[0]*a[0] + k[1]*a[1] + k[2]*a[2])*inv_k2
    for i in range(3):
        a[i] -= k[i]*p


def nonlinear_hat(u_hat):
    ud = cp.stack([bwd(u_hat[i]*dealias) for i in range(3)])
    nl = cp.zeros_like(ud)
    for i, j, l in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        om = bwd(1j*(k[j]*u_hat[l] - k[l]*u_hat[j])*dealias)
        nl[j] += ud[l]*om
        nl[l] -= ud[j]*om
    del ud, om
    r = cp.stack([fwd(nl[i]) for i in range(3)])
    del nl
    project(r)
    return r*dealias


def write(iout, u_hat):
    u = cp.stack([bwd(u_hat[i]) for i in range(3)])[:, :, :H, :]
    out = np.zeros((NZ, NY, NX, 3), np.float32)
    for i, s in enumerate((1, -1, 1)):
        a = cp.asnumpy(u[i]).transpose(2, 1, 0)
        out[:, 1:NY - 1, :, i] = a
        out[:, 0, :, i] = s*a[:, 0, :]
        out[:, NY - 1, :, i] = s*a[:, -1, :]
    with h5py.File(f"fields_{iout:04d}.vtkhdf", "w") as f:
        g = f.create_group("VTKHDF")
        g.attrs["Version"] = np.array([1, 0], np.int32)
        g.attrs.create("Type", np.bytes_("ImageData"))
        g.attrs["WholeExtent"] = np.array([0, NX - 1, 0, NY - 1, 0, NZ - 1], np.int64)
        g.attrs["Origin"] = np.zeros(3)
        g.attrs["Spacing"] = np.ones(3)
        g.attrs["Direction"] = np.eye(3).ravel()
        g.create_group("PointData").create_dataset("velocity", data=out)


mirror = lambda a, s: cp.concatenate([cp.asarray(a, dtype=cp.float64), s*cp.asarray(a[:, ::-1, :], dtype=cp.float64)], axis=1)
u_hat = cp.stack([fwd(mirror(u0[i], s)) for i, s in enumerate((1, -1, 1))])
del u0
project(u_hat)
u_hat *= dealias

A = (0.0, -5/9, -153/128)
B = (1/3, 15/16, 8/15)
dt_visc = 1/(nu*3*kmax**2)
t, step, iout = 0.0, 0, 0
residual = cp.zeros_like(u_hat)
print(f"NX={NX} 2H={2*H} NZ={NZ} nu={nu:.4e} T_REF={T_REF:.2f}", flush=True)

while True:

    if t >= iout*T_REF - 1e-9:
        write(iout, u_hat)
        print(f"out={iout:3d} t/T_REF={t/T_REF:6.2f} step={step}", flush=True)
        iout += 1
        if iout > t_end:
            break

    if step % 5 == 0:
        u = cp.stack([bwd(u_hat[i]) for i in range(3)])
        dt = min(cfl/float(cp.max(cp.abs(u[0]) + cp.abs(u[1]) + cp.abs(u[2]))), dt_visc)
        del u
        if not np.isfinite(dt):
            sys.exit(f"NaN/inf em step={step} t={t:.6f}")

    dt_step = max(min(dt, iout*T_REF - t), 1e-12)

    for a, b in zip(A, B):
        r = nonlinear_hat(u_hat)
        for i in range(3):
            r[i] -= nu*k2*u_hat[i]
            residual[i] *= a
            residual[i] += dt_step*r[i]
            u_hat[i] += b*residual[i]
        del r

    t += dt_step
    step += 1
