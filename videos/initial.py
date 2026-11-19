"""Condicao inicial da camada de mistura temporal para o video dos ribs (variante do ML/initial.py).

   Mesma construcao do caso ROLLUP de Rogers & Moser (1992), com tres mudancas para o video:
   - Lx = 2 lambda_x: dois rolos + subharmonico (1/2,0), que faz os rolos parearem e enrolarem os ribs;
   - Lz = 3 lambda_z: tres pares de ribs ao longo do rolo, em vez de um;
   - modos espanwise extras (kz = 1..6 vezes o fundamental do dominio) com fases aleatorias e
     amplitude pequena: os ribs deixam de ser identicos, como numa onda real.
   Grava initial.h5 (so os 10 campos) no formato lido por shearLayerVideo.hpp, que tem os mesmos parametros no codigo.
   Uso: .venv/bin/python initial.py
"""

import numpy as np
import scipy.linalg as la
from scipy.special import erf
import h5py

LX = 256                       # lambda_x em celulas (delta_w = 35.1, como no ML)
NX, NY, NZ = 2*LX, 512, 3*154
U, RE = 0.05, 1000
A10, A05 = 0.1, 0.1            # amplitudes do fundamental e do subharmonico
GAMMA, NOISE = 0.025, 0.35     # Gamma_x/Gamma_z do modo (0,1); amplitude relativa dos modos extras
rng = np.random.default_rng(7)

DW = LX/(1.16*2*np.pi)
nu = U*DW/RE

y = (np.arange(NY) - 0.5*(NY - 1))/DW
h = 1/DW
f = slice(1, NY - 1)
N = NY - 2
D2 = (np.diag(np.ones(N - 1), 1) - 2*np.eye(N) + np.diag(np.ones(N - 1), -1))/h**2
Ub = erf(np.sqrt(np.pi)*y)
Upp = D2 @ Ub[f]
Upp[[0, -1]] = 0


def orr_sommerfeld(kx, amp):
    """Autofuncao mais instavel, normalizada como em R&M (eq. 7), com v real em y = 0."""
    L = D2 - kx**2*np.eye(N)
    c, V = la.eig(np.diag(Ub[f]) @ L - np.diag(Upp) - (L @ L)/(1j*kx*RE), L)
    k = np.argmax(c.imag)
    phi = np.zeros(NY, complex)
    phi[f] = V[:, k]
    uh, vh = 1j*np.gradient(phi, h)/kx, phi
    A = np.sqrt(np.trapezoid(2*(np.abs(uh/2)**2 + np.abs(vh/2)**2), dx=h))
    phase = np.exp(-1j*np.angle(vh[NY//2]))
    print(f"kx = {kx:.3f}: omega_i = {kx*c[k].imag:.4f} U/delta_w")
    return amp*uh*phase/A, amp*vh*phase/A


x = np.arange(NX)[:, None, None]
z = np.arange(NZ)[None, None, :]
u = U*Ub[None, :, None] + 0*x + 0*z
v = np.zeros_like(u)
for m, amp in ((2, A10), (1, A05)):       # m = numero de onda em x no dominio
    uh, vh = orr_sommerfeld(2*np.pi*m/NX*DW, amp)
    e = np.exp(1j*(2*np.pi*m*x/NX - np.pi))
    u += U*np.real(uh[None, :, None]*e)
    v += U*np.real(vh[None, :, None]*e)

w = np.zeros_like(u)
modes = [(3, GAMMA, 0.0)] + [(n, NOISE*GAMMA, rng.uniform(0, 2*np.pi)) for n in (1, 2, 4, 5, 6)]
for n, gam, ph in modes:
    kz = 2*np.pi*n/NZ*DW
    w0 = gam*2*(LX/DW)*kz/2
    psi = np.zeros(NY)
    psi[f] = np.linalg.solve(D2 - kz**2*np.eye(N), -w0*np.exp(-np.pi*y[f]**2))
    dpsi = np.gradient(psi, h)
    arg = 2*np.pi*n*z/NZ + ph
    v += U*kz*psi[None, :, None]*np.cos(arg)
    w += -U*dpsi[None, :, None]*np.sin(arg)

gx = lambda q: (np.roll(q, -1, 0) - np.roll(q, 1, 0))/2
gy = lambda q: np.gradient(q, axis=1)
gz = lambda q: (np.roll(q, -1, 2) - np.roll(q, 1, 2))/2

fields = {"drho": 0*u, "ux": u, "uy": v, "uz": w,
          "sxx": gx(u), "sxy": 0.5*(gy(u) + gx(v)), "sxz": 0.5*(gz(u) + gx(w)),
          "syy": gy(v), "syz": 0.5*(gz(v) + gy(w)), "szz": gz(w)}

with h5py.File("initial.h5", "w") as out:
    for name, q in fields.items():
        out.create_dataset(name, data=np.asarray(q, dtype=np.float32))

print(f"delta_w = {DW:.3f}, tau = {0.5 + 3*nu:.5f}")
