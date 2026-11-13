"""Estatisticas do LBM com as mesmas definicoes de spectral.py.

   Os blocos ESPACOS E ALOCACAO, DIAGNOSTICO e ESTATISTICAS sao copia literal de
   spectralDNS/spectral.py. So a leitura e o laco sobre os snapshots sao novos.
   Uso, dentro da pasta da corrida: mpirun -np 32 python ../statistics.py
"""

import mpi4py
mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import numpy as np
import h5py
from shenfun import Array, Function, FunctionSpace, TensorProductSpace, VectorSpace

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ======== PARAMETROS FISICOS (lidos da condicao inicial) ========

P = dict(line.split()[:2] for line in open("initial.txt") if line.split() and not line.startswith("#"))
N   = int(P["N"])
nu  = float(P["nu"])
te0 = float(P["te"])

L = 2*np.pi
delta = L/N
kmax = N/3

# ======== ESPACOS E ALOCACAO ========

Vx = FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=np.complex128)
Vy = FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=np.complex128)
Vz = FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=np.float64)
T  = TensorProductSpace(comm, [Vx, Vy, Vz], dtype=np.float64, slab=False, collapse_fourier=True)
VT = VectorSpace(T)
Td  = T.get_dealiased(padding_factor=1, dealias_direct=True)
VTd = VectorSpace(Td)

k = [ki.astype(np.float64, copy=False) for ki in T.local_wavenumbers(scaled=True)]
k2 = np.zeros(T.shape(True), dtype=np.float64)
for ki in k:
    k2 += ki*ki
inv_k2 = np.zeros_like(k2)
np.divide(1, k2, out=inv_k2, where=k2 != 0)
dealias = ((np.abs(k[0]) < N//3) & (np.abs(k[1]) < N//3) & (np.abs(k[2]) < N//3)).astype(np.float64)

u         = Array(VT)
u_dealias = Array(VTd)
omega     = Array(VTd)
nonlinear = Array(VTd)

u_hat     = Function(VT)
omega_hat = Function(VT)
rhs_hat   = Function(VT)
residual  = Function(VT)
pressure_hat = Function(T)

# ======== DIAGNOSTICO ========

Kb = T.local_wavenumbers(scaled=True, broadcast=True, eliminate_highest_freq=False)
Kb = [ki.astype(np.float64, copy=False) for ki in Kb]
kz = k[2]
w = np.broadcast_to(np.where((kz == 0) | (kz == N//2), 1.0, 2.0), k2.shape)
shell = np.rint(np.sqrt(k2)).astype(int)
nshell = N//2 + 1
kc = np.arange(nshell)

# direcoes de analise e suas transversais (n . m = 0)
n_dir = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1]], dtype=float)
m_dir = np.array([[0,1,0], [0,0,1], [1,0,0], [-1,1,0], [1,-1,0]], dtype=float)
n_dir /= np.linalg.norm(n_dir, axis=1)[:, None]
m_dir /= np.linalg.norm(m_dir, axis=1)[:, None]
ndir = len(n_dir)

# separacoes de D_LL: 0 ... L/2, passo 2*delta
r_sep = np.arange(0, N//2 + 1, 2)*delta
nr = len(r_sep)

du     = Array(T)
du_hat = Function(T)
scal   = Array(Td)
tmp    = Function(T)

def shell_sum(field):
    return comm.allreduce(np.bincount(shell.ravel(), weights=(field*w).ravel(),
                                      minlength=nshell+5)[:nshell])

def moments(a):
    return comm.allreduce(np.array([np.sum(a**2), np.sum(a**3), np.sum(a**4)]))/N**3

def nonlinear_hat():
    """termo nao-linear projetado P(u x curl(u)); deixa o resultado em rhs_hat"""
    omega_hat[0] = 1j*(k[1]*u_hat[2] - k[2]*u_hat[1])
    omega_hat[1] = 1j*(k[2]*u_hat[0] - k[0]*u_hat[2])
    omega_hat[2] = 1j*(k[0]*u_hat[1] - k[1]*u_hat[0])
    VTd.backward(u_hat, u_dealias)
    VTd.backward(omega_hat, omega)
    nonlinear[0] = u_dealias[1]*omega[2] - u_dealias[2]*omega[1]
    nonlinear[1] = u_dealias[2]*omega[0] - u_dealias[0]*omega[2]
    nonlinear[2] = u_dealias[0]*omega[1] - u_dealias[1]*omega[0]
    VTd.forward(nonlinear, rhs_hat)
    rhs_hat[:] *= dealias
    pressure_hat[:] = (k[0]*rhs_hat[0] + k[1]*rhs_hat[1] + k[2]*rhs_hat[2])*inv_k2
    for i in range(3):
        rhs_hat[i] -= k[i]*pressure_hat
    return rhs_hat

def rhs():
    """P(u x curl(u)) - nu*k^2*u"""
    nonlinear_hat()
    for i in range(3):
        rhs_hat[i] -= nu*k2*u_hat[i]
    return rhs_hat

def compute_dt():
    VT.backward(u_hat, u)
    umax = comm.allreduce(np.max(np.abs(u)), op=MPI.MAX)
    return cfl*delta/max(umax, 1e-300)

# ======== ESTATISTICAS ========

def statistics():
    """devolve (escalares, E(k), Pi(k), Ep(k), D_LL(r,dir))"""

    # espectro e escalas derivadas
    Ek = shell_sum(0.5*np.sum(np.abs(u_hat)**2, axis=0))
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
    nl = nonlinear_hat()
    Tk = shell_sum(np.sum(np.real(np.conj(u_hat)*nl), axis=0))
    Pi = -np.cumsum(Tk)

    # pressao: -k^2 p = -k_i k_j (u_i u_j)^
    VTd.backward(u_hat, u_dealias)
    pressure_hat[:] = 0
    for i in range(3):
        for j in range(i, 3):
            scal[:] = u_dealias[i]*u_dealias[j]
            Td.forward(scal, tmp)
            pressure_hat[:] -= (2 - (i == j))*k[i]*k[j]*tmp
    pressure_hat[:] *= inv_k2
    Ep = shell_sum(0.5*np.abs(pressure_hat)**2)
    p_var = 2*Ep.sum()

    # derivadas longitudinal e transversal nas 5 direcoes
    S  = np.zeros(ndir); F  = np.zeros(ndir)
    St = np.zeros(ndir); Ft = np.zeros(ndir)
    ratio = np.zeros(ndir)
    DLL = np.zeros((ndir, nr))

    for i, (n, m) in enumerate(zip(n_dir, m_dir)):
        ikn = 1j*(n[0]*Kb[0] + n[1]*Kb[1] + n[2]*Kb[2])
        uL_hat = n[0]*u_hat[0] + n[1]*u_hat[1] + n[2]*u_hat[2]
        uT_hat = m[0]*u_hat[0] + m[1]*u_hat[1] + m[2]*u_hat[2]

        du_hat[:] = ikn*uL_hat
        T.backward(du_hat, du)
        m2, m3, m4 = moments(du)
        S[i], F[i] = m3/m2**1.5, m4/m2**2

        du_hat[:] = ikn*uT_hat
        T.backward(du_hat, du)
        t2, t3, t4 = moments(du)
        St[i], Ft[i] = t3/t2**1.5, t4/t2**2
        ratio[i] = t2/m2                      # = 2 exatamente se isotropico

        # D_LL(r) = 2*[<uL^2> - R_LL(r)], somando E_L por valor de n.k = m/|v|
        EL = (np.abs(uL_hat)**2*w).real
        v = np.rint(n/np.abs(n[n != 0]).min())
        mk = np.rint(v[0]*Kb[0] + v[1]*Kb[1] + v[2]*Kb[2]).astype(int)
        off = int(np.abs(v).sum())*N//2
        Em = comm.allreduce(np.bincount((mk + off).ravel(), weights=EL.ravel(), minlength=2*off + 1))
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

# ======== SNAPSHOTS DO LBM ========

steps = np.loadtxt("times.dat", ndmin=2)[:, 1]
sl = T.local_slice(False)

rec_t, rec_s, rec_E, rec_Pi, rec_Ep, rec_D = [], [], [], [], [], []

if rank == 0:
    with open("statistics.dat", "w") as g:
        g.write("# " + HEADER + "\n")

with h5py.File("fields.h5", "r", driver="mpio", comm=comm) as f:
    for iout, step in enumerate(steps):
        d = f[f"u/3D/{iout}"]
        for c in range(3):
            u[c] = d[(c,) + tuple(sl)]
        VT.forward(u, u_hat)

        t = step*delta
        s, Ek, Pi, Ep, DLL = statistics()
        rec_t.append(t); rec_s.append(s)
        rec_E.append(Ek); rec_Pi.append(Pi); rec_Ep.append(Ep); rec_D.append(DLL)
        if rank == 0:
            with open("statistics.dat", "a") as g:
                g.write(" ".join(f"{v:.10e}" for v in np.concatenate(([t, t/te0], s))) + "\n")
            print(f"out={iout:3d} t/te={t/te0:5.2f} ReL={s[5]:6.1f} "
                  f"ratio={np.mean(s[12+4*ndir:12+5*ndir]):.3f}", flush=True)

if rank == 0:
    np.savez("spectrum.npz", t=np.array(rec_t), t_te=np.array(rec_t)/te0,
             kc=kc, E=np.array(rec_E), Pi=np.array(rec_Pi), Ep=np.array(rec_Ep),
             r=r_sep, DLL=np.array(rec_D), n_dir=n_dir, m_dir=m_dir,
             nu=nu, N=N, te0=te0)
