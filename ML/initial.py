"""Condicao inicial do caso ROLLUP de Rogers & Moser (1992), J. Fluid Mech. 243, 183-226, em unidades de rede.

   Perfil base (eq. 3):      u = U erf(sqrt(pi) y / delta_w)
   Reynolds:                 Re = U delta_w / nu = 500, U = metade da diferenca de velocidade
   Dominio (tabelas 1 e 2):  Lx = lambda_x = 1.16 (2 pi) delta_w,  Lz = 0.6 Lx
   Modo (1,0):               autofuncao viscosa de Orr-Sommerfeld (VE), amplitude A10 = 0.1 (eq. 7)
   Modo (0,1):               omega_x gaussiano (wG), Gamma_x / Gamma_z = 0.025 (eq. 12)
   Paredes de deslizamento livre em j = 1/2 e j = NY - 3/2 (j = 0 e NY - 1 sao linhas fantasma).
   Grava initial.vtkhdf (ImageData) so com drho, u, v, w e S_ij em VTKHDF/PointData, shape [NZ, NY, NX]
   (x varia mais rapido, como as saidas). Os parametros (U, Re, NX, NY, NZ) ficam no codigo, iguais aos de
   problems/shearLayer.hpp e ML/spectral.py.
   Uso: python initial.py
"""

import numpy as np
import scipy.linalg as la
from scipy.special import erf
import h5py

NX, NY, NZ = 256, 768, 154
U, RE, A10, GAMMA = 0.05, 500, 0.1, 0.025

DW = NX/(1.16*2*np.pi)
nu = U*DW/RE
kx, kz = 2*np.pi/NX*DW, 2*np.pi/NZ*DW

y = (np.arange(NY) - 0.5*(NY - 1))/DW
h = 1/DW
f = slice(1, NY - 1)
N = NY - 2
D2 = (np.diag(np.ones(N - 1), 1) - 2*np.eye(N) + np.diag(np.ones(N - 1), -1))/h**2

Ub = erf(np.sqrt(np.pi)*y)
Upp = D2 @ Ub[f]
Upp[[0, -1]] = 0
L = D2 - kx**2*np.eye(N)
c, V = la.eig(np.diag(Ub[f]) @ L - np.diag(Upp) - (L @ L)/(1j*kx*RE), L)
k = np.argmax(c.imag)
phi = np.zeros(NY, complex)
phi[f] = V[:, k]
uh, vh = 1j*np.gradient(phi, h)/kx, phi
A = np.sqrt(np.trapezoid(2*(np.abs(uh/2)**2 + np.abs(vh/2)**2), dx=h))
phase = np.exp(-1j*np.angle(vh[NY//2]))
uh, vh = A10*uh*phase/A, A10*vh*phase/A

w0 = GAMMA*2*(NX/DW)*kz/2
psi = np.zeros(NY)
psi[f] = np.linalg.solve(D2 - kz**2*np.eye(N), -w0*np.exp(-np.pi*y[f]**2))
dpsi = np.gradient(psi, h)

x = np.arange(NX)[:, None, None]
z = np.arange(NZ)[None, None, :]
e = np.exp(1j*(2*np.pi*x/NX - np.pi))
u = U*(Ub[None, :, None] + np.real(uh[None, :, None]*e)) + 0*z
v = U*(np.real(vh[None, :, None]*e) + kz*psi[None, :, None]*np.cos(2*np.pi*z/NZ))
w = np.broadcast_to(-U*dpsi[None, :, None]*np.sin(2*np.pi*z/NZ), u.shape)

gx = lambda q: (np.roll(q, -1, 0) - np.roll(q, 1, 0))/2
gy = lambda q: np.gradient(q, axis=1)
gz = lambda q: (np.roll(q, -1, 2) - np.roll(q, 1, 2))/2

fields = {"drho": 0*u, "ux": u, "uy": v, "uz": w,
          "sxx": gx(u), "sxy": 0.5*(gy(u) + gx(v)), "sxz": 0.5*(gz(u) + gx(w)),
          "syy": gy(v), "syz": 0.5*(gz(v) + gy(w)), "szz": gz(w)}

with h5py.File("initial.vtkhdf", "w") as out:
    g = out.create_group("VTKHDF")
    g.attrs["Version"] = np.array([1, 0], np.int32)
    g.attrs.create("Type", np.bytes_("ImageData"))
    g.attrs["WholeExtent"] = np.array([0, NX - 1, 0, NY - 1, 0, NZ - 1], np.int64)
    g.attrs["Origin"] = np.zeros(3)
    g.attrs["Spacing"] = np.ones(3)
    g.attrs["Direction"] = np.eye(3).ravel()
    point = g.create_group("PointData")
    for name, q in fields.items():
        point.create_dataset(name, data=np.ascontiguousarray(np.asarray(q, dtype=np.float32).transpose(2, 1, 0)))

print(f"delta_w = {DW:.3f}, tau = {0.5 + 3*nu:.5f}, omega_i = {kx*c[k].imag:.4f} U/delta_w")
