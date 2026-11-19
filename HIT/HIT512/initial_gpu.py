import os
import sys
import numpy as np
import cupy as cp
import h5py

N = 64
TAU = 0.51
ETA = 1.1
TE  = 60

K_FORCE = 2.5

T_SPINUP = 15
T_SPACE = 10
N_REAL = 8

CFL = 0.6

delta = 2*np.pi/N
nu = (TAU - 0.5)/3*delta
eps = nu**3/(ETA*delta)**4

fwd = lambda a: cp.fft.rfftn(a, norm="forward")
bwd = lambda a: cp.fft.irfftn(a, s=(N, N, N), norm="forward")

kx = cp.fft.fftfreq(N, 1/N)
kz = cp.fft.rfftfreq(N, 1/N)

k = [kx[:, None, None], kx[None, :, None], kz[None, None, :]]
k2 = k[0]**2 + k[1]**2 + k[2]**2

inv_k2 = cp.where(k2 != 0, 1/cp.where(k2 != 0, k2, 1), 0)

dealias = (cp.abs(k[0]) < N//3) & (cp.abs(k[1]) < N//3) & (cp.abs(k[2]) < N//3)

w = cp.where((kz == 0) | (kz == N//2), 1.0, 2.0)[None, None, :]
k2_forced = cp.where((k2 > 0) & (k2 < K_FORCE**2), k2, 0.0)


def project(a):
    p = (k[0]*a[0] + k[1]*a[1] + k[2]*a[2])*inv_k2
    for i in range(3):
        a[i] -= k[i]*p


def curl(i, j):
    c = k[i]*u_hat[j]
    c -= k[j]*u_hat[i]
    c *= dealias
    c *= 1j
    return bwd(c)


def nonlinear():
    ud = cp.stack([bwd(u_hat[i]*dealias) for i in range(3)])
    nl = cp.zeros_like(ud)
    om = curl(1, 2)
    nl[1] += ud[2]*om
    nl[2] -= ud[1]*om
    om = curl(2, 0)
    nl[0] -= ud[2]*om
    nl[2] += ud[0]*om
    om = curl(0, 1)
    nl[0] += ud[1]*om
    nl[1] -= ud[0]*om
    del om, ud
    r = cp.stack([fwd(nl[i]) for i in range(3)])
    del nl
    r *= dealias
    project(r)
    return r


def energy():
    return 0.5*cp.sum(cp.abs(u_hat)**2, axis=0)*w


def save(name):
    e = energy()
    K, Om, Lk = float(cp.sum(e)), float(cp.sum(e*k2)), float(cp.sum(e*cp.sqrt(inv_k2)))
    urms = np.sqrt(2*K/3)
    lamb = np.sqrt(15*urms**2/(2*Om))
    eta = (nu**3/(2*nu*Om))**0.25
    L11 = 3*np.pi/4*Lk/K
    u = cp.stack([bwd(u_hat[i]) for i in range(3)])
    umax = float(cp.sqrt(cp.max(u[0]**2 + u[1]**2 + u[2]**2)))
    with h5py.File(name + ".tmp", "w") as f:
        g = f.create_group("VTKHDF")
        g.attrs["Version"] = np.array([1, 0], np.int32)
        g.attrs.create("Type", np.bytes_("ImageData"))
        g.attrs["WholeExtent"] = np.array([0, N - 1, 0, N - 1, 0, N - 1], np.int64)
        g.attrs["Origin"] = np.zeros(3)
        g.attrs["Spacing"] = np.full(3, delta)
        g.attrs["Direction"] = np.eye(3).ravel()
        p = g.create_group("PointData")
        put = lambda c, a: p.create_dataset(c, data=cp.ascontiguousarray(a.T).get())
        for i, c in enumerate(("ux", "uy", "uz")):
            put(c, u[i])
        del u
        ud = cp.stack([bwd(u_hat[i]*dealias) for i in range(3)])
        ph = cp.zeros_like(u_hat[0])
        for i in range(3):
            for j in range(3):
                ph -= k[i]*k[j]*fwd(ud[i]*ud[j])
        del ud
        put("drho", 3*bwd(ph*inv_k2))
        del ph
        for c, (i, j) in {"sxx": (0, 0), "syy": (1, 1), "szz": (2, 2), "sxy": (0, 1), "sxz": (0, 2), "syz": (1, 2)}.items():
            put(c, bwd(0.5j*(k[j]*u_hat[i] + k[i]*u_hat[j])))
    os.rename(name + ".tmp", name)
    print(f"SNAPSHOT {name} t/te={t/TE:.2f} ReL={urms*lamb/nu:.1f} eta/delta={eta/delta:.3f} urms={urms:.6f} "
          f"L11={L11:.3f} te={L11/urms:.2f} Ma_rms={np.sqrt(3)*urms:.4f} Ma_max={np.sqrt(3)*umax:.4f}", flush=True)


rng = cp.random.default_rng()
u_hat = cp.stack([fwd(rng.standard_normal((N, N, N)))*cp.exp(-k2/K_FORCE**2) for _ in range(3)])
project(u_hat)
u_hat *= (2*eps)**(1/3)/np.sqrt(2*float(cp.sum(energy()))/3)

t_save = TE*(T_SPINUP + T_SPACE*np.arange(N_REAL))
t, step, i = 0.0, 0, 0
residual = cp.zeros_like(u_hat)
print(f"nu={nu:.6e} eps={eps:.4e} tau={TAU} colheitas em t/te = {t_save/TE}", flush=True)

while i < N_REAL:
    if step % 5 == 0:
        umax = max(float(cp.max(cp.abs(bwd(u_hat[j])))) for j in range(3))
        dt = min(CFL*delta/umax, 1/(3*nu*(N/3)**2))
        if not np.isfinite(dt):
            sys.exit(f"NaN/inf em step={step} t={t:.6f}")
        nu_f = eps/(2*float(cp.sum(energy()*k2_forced)))

    dt_step = max(min(dt, t_save[i] - t), 1e-12)
    for a, b in zip((0.0, -5/9, -153/128), (1/3, 15/16, 8/15)):
        r = nonlinear()
        L = nu_f*k2_forced - nu*k2
        for j in range(3):
            r[j] += L*u_hat[j]
            residual[j] *= a
            residual[j] += dt_step*r[j]
            u_hat[j] += b*residual[j]
        del r, L
    t += dt_step
    step += 1

    if step % 100 == 0:
        print(f"step={step} t/te={t/TE:.2f} nu_f/nu={nu_f/nu:.2f} dt={dt:.3e}", flush=True)

    if t >= t_save[i] - 1e-12:
        residual = None
        save(f"initial_{i}.vtkhdf")
        residual = cp.zeros_like(u_hat)
        i += 1
