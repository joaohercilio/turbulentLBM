import re
import sympy as sp
from sympy.printing.cxx import cxxcode

cx = [0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 0, 0, 0, 0, -1, 1, 1, -1, 1, -1, 1, -1]
cy = [0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1]
cz = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1]
W = {0: sp.Rational(8, 27), 1: sp.Rational(2, 27), 2: sp.Rational(1, 54), 3: sp.Rational(1, 216)}

Q = 27
c = list(zip(cx, cy, cz))
w = [W[sum(x*x for x in c[i])] for i in range(Q)]
cs2 = sp.Rational(1, 3)
R = range(3)

drho, rho, uu = sp.symbols("drho rho uu")
j = sp.symbols("jx jy jz")
u = sp.symbols("ux uy uz")
pxx, pxy, pxz, pyy, pyz, pzz = sp.symbols("pxx pxy pxz pyy pyz pzz")
p = [[pxx, pxy, pxz],
     [pxy, pyy, pyz],
     [pxz, pyz, pzz]]


def a0():
    return drho

def a1(a):
    return rho*u[a]

def a2(a, b):
    return p[a][b]

def a3(a, b, g):
    return u[a]*a2(b, g) + u[b]*a2(a, g) + u[g]*a2(a, b) - 2*rho*u[a]*u[b]*u[g]

def H1(ci, a):
    return ci[a]

def H2(ci, a, b):
    return ci[a]*ci[b] - cs2*int(a == b)

def H3(ci, a, b, g):
    return ci[a]*ci[b]*ci[g] - cs2*(ci[a]*int(b == g) + ci[b]*int(a == g) + ci[g]*int(a == b))


f37 = [w[i]*(a0()
             + sum(H1(c[i], a)*a1(a) for a in R)/cs2
             + sum(H2(c[i], a, b)*a2(a, b) for a in R for b in R)/(2*cs2**2)
             + sum(H3(c[i], a, b, g)*a3(a, b, g) for a in R for b in R for g in R)/(6*cs2**3))
       for i in range(Q)]

cu  = [sum(c[i][a]*u[a] for a in R) for i in range(Q)]
cj  = [sum(c[i][a]*j[a] for a in R) for i in range(Q)]
cpu = [sum(c[i][a]*p[a][b]*u[b] for a in R for b in R) for i in range(Q)]
A   = [sum(H2(c[i], a, b)*a2(a, b) for a in R for b in R)/(2*cs2**2) for i in range(Q)]

f = [w[i]*(drho + cj[i]/cs2 + A[i]*(1 + cu[i]/cs2) - cpu[i]/cs2**2 + cj[i]*(uu - cu[i]**2)/cs2**2)
     for i in range(Q)]

sub = {uu: sum(x**2 for x in u), **{j[a]: rho*u[a] for a in R}}
for i in range(Q):
    assert sp.expand(f[i].subs(sub) - f37[i]) == 0


def code(e):
    s = cxxcode(e, standard="c++17").replace(".0", ".0f")
    return re.sub(r"std::pow\(([^()]+), 2\)", r"((\1)*(\1))", s)


print("const dfloat rho = drho + RHO_0;")
print("const dfloat invRho = 1.0f / rho;")
print("const dfloat ux = jx * invRho;")
print("const dfloat uy = jy * invRho;")
print("const dfloat uz = jz * invRho;")
print("const dfloat uu = ux*ux + uy*uy + uz*uz;")
print("switch (k) {")
for k in range(Q):
    print(f"    case {k}: return {code(f[k])};")
print("}")
