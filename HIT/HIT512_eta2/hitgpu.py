"""Pseudo-espectral de THI na GPU (CuPy): o que spectral.py e initial.py fazem.

   Convencoes do shenfun (conferidas contra o shenfun): forward = rfftn/N^3,
   backward = irfftn*N^3, dealias = |k_i| < N//3, k com o modo de Nyquist.
   nonlinear_hat, compute_dt, diagnostics e statistics reproduzem, expressao por
   expressao, as funcoes homonimas de spectralDNS/spectral.py e initial.py.
   init(N, nu) antes de usar.
"""

import os
import numpy as np
import cupy as cp

fwd = lambda a: cp.fft.rfftn(a, norm="forward")


def params_path(path):
    """initial_3.vtkhdf -> initial_3.txt: parametros da IC, fora do VTKHDF"""
    return os.path.splitext(path)[0] + ".txt"


def read_params(path):
    """{nome: valor} de um arquivo "nome valor" por linha (int, float ou texto)"""
    P = {}
    for line in open(path):
        line = line.split("#")[0].split(None, 1)
        if len(line) == 2:
            v = line[1].strip()
            for conv in (int, float, str):
                try:
                    P[line[0]] = conv(v)
                    break
                except ValueError:
                    pass
    return P


def write_params(path, P):
    with open(path, "w") as g:
        for key, val in P.items():
            val = val.item() if hasattr(val, "item") else val
            g.write(f"{key} {val!r}\n" if isinstance(val, float) else f"{key} {val}\n")


def init(N_, nu_):
    global N, nu, delta, kmax, bwd, k, k2, inv_k2, inv_k, dealias, w, shell, nshell, kc
    global n_dir, m_dir, ndir, r_sep, nr
    N, nu = N_, nu_
    delta = 2*np.pi/N
    kmax = N/3
    bwd = lambda a: cp.fft.irfftn(a, s=(N, N, N), norm="forward")

    kx = cp.fft.fftfreq(N, 1/N)
    kz = cp.fft.rfftfreq(N, 1/N)
    k = [kx[:, None, None], kx[None, :, None], kz[None, None, :]]
    k2 = k[0]**2 + k[1]**2 + k[2]**2
    inv_k2 = cp.where(k2 != 0, 1/cp.where(k2 != 0, k2, 1), 0)
    inv_k = cp.sqrt(inv_k2)
    dealias = (cp.abs(k[0]) < N//3) & (cp.abs(k[1]) < N//3) & (cp.abs(k[2]) < N//3)

    w = cp.where((kz == 0) | (kz == N//2), 1.0, 2.0)[None, None, :]
    shell = cp.rint(cp.sqrt(k2)).astype(cp.int64).ravel()
    nshell = N//2 + 1
    kc = np.arange(nshell)

    n_dir = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1]], dtype=float)
    m_dir = np.array([[0,1,0], [0,0,1], [1,0,0], [-1,1,0], [1,-1,0]], dtype=float)
    n_dir /= np.linalg.norm(n_dir, axis=1)[:, None]
    m_dir /= np.linalg.norm(m_dir, axis=1)[:, None]
    ndir = len(n_dir)

    r_sep = np.arange(0, N//2 + 1, 2)*delta
    nr = len(r_sep)


def read_velocity(dset):
    """u_hat de um dataset (3,N,N,N) ou de tres datasets (ux, uy, uz), indice [x, y, z]"""
    return cp.stack([fwd(cp.asarray(d[:]).astype(cp.float64)) for d in dset])


def read_initial(f):
    """u_hat de uma IC em VTKHDF (PointData com shape [z, y, x])"""
    return cp.stack([fwd(cp.ascontiguousarray(cp.asarray(f["VTKHDF/PointData/" + c][:]).astype(cp.float64).T))
                     for c in ("ux", "uy", "uz")])


def vtkhdf_point(f, name, a):
    """grava o campo a[x, y, z] (CuPy) em VTKHDF/PointData/name com shape [z, y, x]"""
    f["VTKHDF/PointData"].create_dataset(name, data=cp.ascontiguousarray(a.T).get())


def shell_sum(field):
    return cp.bincount(shell, weights=cp.broadcast_to(field*w, k2.shape).ravel(),
                       minlength=nshell+5)[:nshell].get()


def moments(a):
    return np.array([float(cp.sum(a**2)), float(cp.sum(a**3)), float(cp.sum(a**4))])/N**3


def curl(u_hat, i, j):
    """componente fisica 1j*(k_i u_j - k_j u_i)*dealias do rotacional, com temporarios in-place"""
    c = k[i]*u_hat[j]
    c -= k[j]*u_hat[i]
    c *= dealias
    c *= 1j
    return bwd(c)


def nonlinear_hat(u_hat, keep_ud=False):
    """termo nao-linear projetado P(u x curl(u)); devolve rhs_hat (e u_dealias)
       rotacional uma componente por vez: mesmo resultado bit a bit, menos memoria"""
    ud = cp.stack([bwd(u_hat[i]*dealias) for i in range(3)])
    nl = cp.zeros_like(ud)
    om = curl(u_hat, 1, 2)
    nl[1] += ud[2]*om
    nl[2] -= ud[1]*om
    om = curl(u_hat, 2, 0)
    nl[0] -= ud[2]*om
    nl[2] += ud[0]*om
    om = curl(u_hat, 0, 1)
    nl[0] += ud[1]*om
    nl[1] -= ud[0]*om
    del om
    if not keep_ud:
        del ud
    rhs_hat = cp.empty((3,) + k2.shape, dtype=cp.complex128)
    for i in range(3):
        rhs_hat[i] = fwd(nl[i])
    del nl
    rhs_hat *= dealias
    pressure_hat = (k[0]*rhs_hat[0] + k[1]*rhs_hat[1] + k[2]*rhs_hat[2])*inv_k2
    for i in range(3):
        rhs_hat[i] -= k[i]*pressure_hat
    return (rhs_hat, ud) if keep_ud else rhs_hat


def compute_dt(u_hat, cfl):
    umax = max(float(cp.max(cp.abs(bwd(u_hat[i])))) for i in range(3))
    return cfl*delta/max(umax, 1e-300)


def diagnostics(u_hat):
    """initial.py: quantidades turbulentas somando modo a modo com |k| exato"""
    e = 0.5*cp.sum(cp.abs(u_hat)**2, axis=0)*w
    Km, Om, Lk = float(cp.sum(e)), float(cp.sum(e*k2)), float(cp.sum(e*inv_k))
    ep  = 2*nu*Om
    ur  = np.sqrt(2*Km/3)
    lm  = np.sqrt(15*ur**2/(2*Om)) if Om > 0 else 0.0
    et  = (nu**3/ep)**0.25 if ep > 0 else 0.0
    l11 = 3*np.pi/4*Lk/Km if Km > 0 else 0.0
    return dict(K=Km, eps=ep, urms=ur, lamb=lm, ReL=ur*lm/nu, eta=et, L11=l11,
                te=l11/ur if ur > 0 else 0.0, kmax_eta=et*kmax, eta_delta=et/delta)


def statistics(u_hat):
    """spectral.py: devolve (escalares, E(k), Pi(k), Ep(k), D_LL(r,dir))"""

    # espectro e escalas derivadas
    Ek = shell_sum(0.5*cp.sum(cp.abs(u_hat)**2, axis=0))
    K   = Ek.sum()
    Om  = (Ek*kc**2).sum()
    eps = 2*nu*Om
    urms = np.sqrt(2*K/3)
    lamb = np.sqrt(15*urms**2/(2*Om)) if Om > 0 else 0.0
    eta  = (nu**3/eps)**0.25 if eps > 0 else 0.0
    L11  = 3*np.pi/4*np.sum(Ek[1:]/kc[1:])/K if K > 0 else 0.0
    ReL  = urms*lamb/nu
    te   = L11/urms if urms > 0 else 0.0

    # transferencia e fluxo espectral
    nl, ud = nonlinear_hat(u_hat, keep_ud=True)
    Tk = shell_sum(cp.sum(cp.real(cp.conj(u_hat)*nl), axis=0))
    Pi = -np.cumsum(Tk)
    del nl

    # pressao: -k^2 p = -k_i k_j (u_i u_j)^
    pressure_hat = cp.zeros_like(u_hat[0])
    for i in range(3):
        for j in range(i, 3):
            pressure_hat -= (2 - (i == j))*k[i]*k[j]*fwd(ud[i]*ud[j])
    pressure_hat *= inv_k2
    Ep = shell_sum(0.5*cp.abs(pressure_hat)**2)
    p_var = 2*Ep.sum()
    del ud, pressure_hat

    # derivadas longitudinal e transversal nas 5 direcoes
    S  = np.zeros(ndir); F  = np.zeros(ndir)
    St = np.zeros(ndir); Ft = np.zeros(ndir)
    ratio = np.zeros(ndir)
    DLL = np.zeros((ndir, nr))

    for i, (n, m) in enumerate(zip(n_dir, m_dir)):
        ikn = 1j*(n[0]*k[0] + n[1]*k[1] + n[2]*k[2])
        uL_hat = n[0]*u_hat[0] + n[1]*u_hat[1] + n[2]*u_hat[2]
        uT_hat = m[0]*u_hat[0] + m[1]*u_hat[1] + m[2]*u_hat[2]

        m2, m3, m4 = moments(bwd(ikn*uL_hat))
        S[i], F[i] = m3/m2**1.5, m4/m2**2

        t2, t3, t4 = moments(bwd(ikn*uT_hat))
        St[i], Ft[i] = t3/t2**1.5, t4/t2**2
        ratio[i] = t2/m2                      # = 2 exatamente se isotropico

        # D_LL(r) = 2*[<uL^2> - R_LL(r)], somando E_L por valor de n.k = m/|v|
        EL = cp.abs(uL_hat)**2*w
        v = np.rint(n/np.abs(n[n != 0]).min())
        mk = cp.rint(v[0]*k[0] + v[1]*k[1] + v[2]*k[2]).astype(cp.int64)
        off = int(np.abs(v).sum())*N//2
        Em = cp.bincount(cp.broadcast_to(mk + off, k2.shape).ravel(), weights=EL.ravel(),
                         minlength=2*off + 1).get()
        RLL = np.cos(np.outer(r_sep, (np.arange(2*off + 1) - off)/np.linalg.norm(v))) @ Em
        DLL[i] = 2*(RLL[0] - RLL)

    scalars = np.concatenate(([K, eps, Om, urms, lamb, ReL, eta, L11, te,
                               eta*kmax, eta/delta, p_var], S, F, St, Ft, ratio))
    return scalars, Ek, Pi, Ep, DLL


HEADER = ("t t_te K eps Om urms lambda ReL eta L11 te kmax_eta eta_delta p_var "
          + " ".join(f"S{d}"     for d in ("100","010","001","110","111"))
          + " " + " ".join(f"F{d}"     for d in ("100","010","001","110","111"))
          + " " + " ".join(f"St{d}"    for d in ("100","010","001","110","111"))
          + " " + " ".join(f"Ft{d}"    for d in ("100","010","001","110","111"))
          + " " + " ".join(f"ratio{d}" for d in ("100","010","001","110","111")))


class Recorder:
    """grava statistics.dat linha a linha e spectrum.npz no fim, como spectral.py"""

    def __init__(self, te0):
        self.te0 = te0
        self.t, self.s, self.E, self.Pi, self.Ep, self.D = [], [], [], [], [], []
        with open("statistics.dat", "w") as g:
            g.write("# " + HEADER + "\n")

    def add(self, t, u_hat):
        s, Ek, Pi, Ep, DLL = statistics(u_hat)
        self.t.append(t); self.s.append(s)
        self.E.append(Ek); self.Pi.append(Pi); self.Ep.append(Ep); self.D.append(DLL)
        with open("statistics.dat", "a") as g:
            g.write(" ".join(f"{v:.10e}" for v in np.concatenate(([t, t/self.te0], s))) + "\n")
        return s

    def save(self):
        np.savez("spectrum.npz", t=np.array(self.t), t_te=np.array(self.t)/self.te0,
                 kc=kc, E=np.array(self.E), Pi=np.array(self.Pi), Ep=np.array(self.Ep),
                 r=r_sep, DLL=np.array(self.D), n_dir=n_dir, m_dir=m_dir,
                 nu=nu, N=N, te0=self.te0)


def harvest(path, u_hat, attrs, t_decay=10):
    """initial.py: escreve u, drho = 3p e S_ij com a = 1 (u_hat ja nas unidades reescaladas)
       em VTKHDF (ImageData, so os 10 campos em PointData com shape [z, y, x]);
       parametros em params_path(path), "nome valor" por linha"""
    import h5py
    cs = 1/np.sqrt(3)
    d = diagnostics(u_hat)
    u = cp.stack([bwd(u_hat[i]) for i in range(3)])
    umax = float(cp.sqrt(cp.max(u[0]**2 + u[1]**2 + u[2]**2)))
    lbm = dict(Ma_rms=d["urms"]/cs, Ma_max=umax/cs, nu_lbm=nu/delta,
               tau=3*nu/delta + 0.5, te_lbm=d["te"]/delta,
               nsteps=int(np.ceil(t_decay*d["te"]/delta)))
    with h5py.File(path, "w") as f:
        g = f.create_group("VTKHDF")
        g.attrs["Version"] = np.array([1, 0], np.int32)
        g.attrs.create("Type", np.bytes_("ImageData"))
        g.attrs["WholeExtent"] = np.array([0, N - 1, 0, N - 1, 0, N - 1], np.int64)
        g.attrs["Origin"] = np.zeros(3)
        g.attrs["Spacing"] = np.full(3, delta)
        g.attrs["Direction"] = np.eye(3).ravel()
        g.create_group("PointData")
        for c, name in enumerate(("ux", "uy", "uz")):
            vtkhdf_point(f, name, u[c])
        del u
        ud = cp.stack([bwd(u_hat[i]*dealias) for i in range(3)])
        pressure_hat = cp.zeros_like(u_hat[0])
        for i in range(3):
            for j in range(3):
                pressure_hat -= k[i]*k[j]*fwd(ud[i]*ud[j])
        pressure_hat *= inv_k2
        del ud
        vtkhdf_point(f, "drho", 3.0*bwd(pressure_hat))
        del pressure_hat
        for name, (i, j) in {"sxx": (0, 0), "syy": (1, 1), "szz": (2, 2),
                             "sxy": (0, 1), "sxz": (0, 2), "syz": (1, 2)}.items():
            vtkhdf_point(f, name, bwd(0.5j*(k[j]*u_hat[i] + k[i]*u_hat[j])))
    write_params(params_path(path), dict(list(attrs.items()) + [("N", N), ("nu", nu), ("cs", cs)]
                                         + list(d.items()) + list(lbm.items())))
    return d, lbm
