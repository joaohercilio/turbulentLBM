import sys
import numpy as np
import cupy as cp
import h5py

cp.cuda.Device(int(sys.argv[1])).use()

N, TAU, TE = 512, 0.51, 60.0

delta = 2*np.pi/N
nu = (TAU - 0.5)/3*delta

fwd = lambda a: cp.fft.rfftn(a, norm="forward")
bwd = lambda a: cp.fft.irfftn(a, s=(N, N, N), norm="forward")
kx, kz = cp.fft.fftfreq(N, 1/N), cp.fft.rfftfreq(N, 1/N)
k = [kx[:, None, None], kx[None, :, None], kz[None, None, :]]
k2 = k[0]**2 + k[1]**2 + k[2]**2
inv_k2 = cp.where(k2 != 0, 1/cp.where(k2 != 0, k2, 1), 0)
dealias = (cp.abs(k[0]) < N//3) & (cp.abs(k[1]) < N//3) & (cp.abs(k[2]) < N//3)
w = cp.where((kz == 0) | (kz == N//2), 1.0, 2.0)[None, None, :]
shell = cp.rint(cp.sqrt(k2)).astype(cp.int64).ravel()
kc = np.arange(N//2 + 1)
r_sep = np.arange(0, N//2 + 1, 2)*delta
n_dir = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1]])/np.sqrt([[1], [1], [1], [2], [3]])
m_dir = np.array([[0,1,0], [0,0,1], [1,0,0], [-1,1,0], [1,-1,0]])/np.sqrt([[1], [1], [1], [2], [2]])
DIRS = ("100", "010", "001", "110", "111")
HEADER = " ".join(["t t_te K eps urms lambda ReL eta L11 te eta_delta"] + [q + d for q in ("S", "F", "ratio") for d in DIRS])


def shell_sum(a):
    return cp.bincount(shell, weights=cp.broadcast_to(a*w, k2.shape).ravel(), minlength=len(kc) + 5)[:len(kc)].get()


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


def record(t):
    E = shell_sum(0.5*cp.sum(cp.abs(u_hat)**2, axis=0))
    K, Om = E.sum(), (E*kc**2).sum()
    urms = np.sqrt(2*K/3)
    lamb = np.sqrt(15*urms**2/(2*Om))
    eta = (nu**3/(2*nu*Om))**0.25
    L11 = 3*np.pi/4*np.sum(E[1:]/kc[1:])/K
    Pi = -np.cumsum(shell_sum(cp.sum(cp.real(cp.conj(u_hat)*nonlinear()), axis=0)))
    S, F, R, D = [], [], [], []
    for n, m in zip(n_dir, m_dir):
        ikn = 1j*(n[0]*k[0] + n[1]*k[1] + n[2]*k[2])
        uL = n[0]*u_hat[0] + n[1]*u_hat[1] + n[2]*u_hat[2]
        dL = bwd(ikn*uL)
        m2, m3, m4 = (float(cp.mean(dL**p)) for p in (2, 3, 4))
        S.append(m3/m2**1.5)
        F.append(m4/m2**2)
        R.append(float(cp.mean(bwd(ikn*(m[0]*u_hat[0] + m[1]*u_hat[1] + m[2]*u_hat[2]))**2))/m2)
        v = np.rint(n/np.abs(n[n != 0]).min())
        off = int(np.abs(v).sum())*N//2
        mk = cp.rint(v[0]*k[0] + v[1]*k[1] + v[2]*k[2]).astype(cp.int64) + off
        Em = cp.bincount(cp.broadcast_to(mk, k2.shape).ravel(), weights=(cp.abs(uL)**2*w).ravel(), minlength=2*off + 1).get()
        RLL = np.cos(np.outer(r_sep, (np.arange(2*off + 1) - off)/np.linalg.norm(v))) @ Em
        D.append(2*(RLL[0] - RLL))
    s = [t, t/TE, K, 2*nu*Om, urms, lamb, urms*lamb/nu, eta, L11, L11/urms, eta/delta] + S + F + R
    out.write(" ".join(f"{x:.10e}" for x in s) + "\n")
    out.flush()
    for a, x in zip(spectra.values(), (t, E, Pi, D)):
        a.append(x)
    print(f"t/te={t/TE:5.2f} ReL={urms*lamb/nu:6.1f} ratio={np.mean(R):.3f}", flush=True)


out = open("statistics.dat", "w")
out.write("# " + HEADER + "\n")
spectra = dict(t=[], E=[], Pi=[], DLL=[])

with h5py.File("fields.h5", "r") as f:
    b = float(f.attrs["scale"])
    for i, step in enumerate(np.loadtxt("times.dat", ndmin=2)[:, 1]):
        u_hat = cp.stack([fwd(cp.asarray(c[:]).astype(cp.float64)) for c in f[f"u/3D/{i}"]])/b
        record(b*step*delta)

np.savez("spectrum.npz", t_te=np.array(spectra["t"])/TE, kc=kc, r=r_sep, N=N, nu=nu,
         **{key: np.array(v) for key, v in spectra.items()})
