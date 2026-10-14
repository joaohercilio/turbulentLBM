"""3-D incompressible Taylor-Green pseudo-spectral solver"""



import mpi4py
mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import numpy as np
from shenfun import Array, Function, FunctionSpace, TensorProductSpace, VectorSpace


N = 256

real     = np.float32   # np.float64
complex_ = np.complex64 # np.complex128

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# periodic fourier space, the last direction is real-to-complex
V = [FunctionSpace(N, "F", domain=(0, 2*np.pi), dtype=real if i == 2 else complex_) for i in range(3)]
T = TensorProductSpace(comm, V, dtype=real, slab=True, collapse_fourier=True)
VT = VectorSpace(T)

# same-sized transform space that truncates the upper 1/3 of modes
Td = T.get_dealiased(padding_factor=1, dealias_direct=True)
VTd = VectorSpace(Td)

x = T.local_mesh(True)
k = T.local_wavenumbers(scaled=True)
k = [ki.astype(real, copy=False) for ki in k]
k2 = np.zeros(T.shape(True), dtype=real)
for ki in k:
    k2 += ki*ki
inv_k2 = np.zeros_like(k2)
np.divide(1, k2, out=inv_k2, where=k2 != 0)

# allocate arrays
u = Array(VT)
u_dealias = Array(VTd)
omega = Array(VTd)
nonlinear = Array(VTd)
u_hat = Function(VT)
omega_hat = Function(VT)
rhs_hat = Function(VT)
residual = Function(VT)
pressure_hat = Function(T)

# taylor green 
#u[0] =  np.sin(x[0])*np.cos(x[1])*np.cos(x[2])
#u[1] = -np.cos(x[0])*np.sin(x[1])*np.cos(x[2])
#u[2] = 0

import vtk
from vtk.util import numpy_support

reader = vtk.vtkStructuredPointsReader()
reader.SetFileName("rogallo/initial.vtk")
reader.Update()

data = reader.GetOutput().GetPointData()

sl = T.local_slice(False)

ux = numpy_support.vtk_to_numpy(data.GetArray("ux")).reshape((N, N, N), order="F")
uy = numpy_support.vtk_to_numpy(data.GetArray("uy")).reshape((N, N, N), order="F")
uz = numpy_support.vtk_to_numpy(data.GetArray("uz")).reshape((N, N, N), order="F")

u[0] = ux[sl]
u[1] = uy[sl]
u[2] = uz[sl]

cfl = 0.6
dx = 2*np.pi/N
nu = 0.0001339440322141572
t_total = 30 * 318

VT.forward(u, u_hat)

def compute_dt():
    VT.backward(u_hat, u)
    umax = comm.allreduce(np.max(np.abs(u)), op=MPI.MAX)
    return cfl*dx/umax

dt = compute_dt()


def rhs():
    """ Compute P(u x curl(u)) - nu*k^2*u in Fourier space """
    omega_hat[0] = 1j*(k[1]*u_hat[2] - k[2]*u_hat[1])
    omega_hat[1] = 1j*(k[2]*u_hat[0] - k[0]*u_hat[2])
    omega_hat[2] = 1j*(k[0]*u_hat[1] - k[1]*u_hat[0])

    VTd.backward(u_hat, u_dealias)
    VTd.backward(omega_hat, omega)

    nonlinear[0] = u_dealias[1]*omega[2] - u_dealias[2]*omega[1]
    nonlinear[1] = u_dealias[2]*omega[0] - u_dealias[0]*omega[2]
    nonlinear[2] = u_dealias[0]*omega[1] - u_dealias[1]*omega[0]
    VTd.forward(nonlinear, rhs_hat)

    # Helmholtz projection: P(a) = a - k*(k.a)/|k|^2
    pressure_hat[:] = (k[0]*rhs_hat[0] + k[1]*rhs_hat[1] + k[2]*rhs_hat[2])*inv_k2
    for i in range(3):
        rhs_hat[i] -= k[i]*pressure_hat
        rhs_hat[i] -= nu*k2*u_hat[i]
    return rhs_hat


def energy():
    VT.backward(u_hat, u)
    local = np.sum(u*u, dtype=np.float64)
    return 0.5*comm.allreduce(local)/N**3


A = (0.0, -5/9, -153/128)
B = (1/3, 15/16, 8/15)
t = 0.0
step = 0
ek = energy()

if rank == 0:
    print("t\tk")
    print(f"{t:.6f}\t{ek:.10e}")

while t < t_total - 1e-14:
    dt = min(dt, t_total - t)

    for a, b in zip(A, B):
        residual *= a
        residual += dt*rhs()
        u_hat += b*residual

    t += dt
    step += 1

    if step % 10 == 0 or t >= t_total - 1e-14:
        ek = energy()
        if rank == 0:
            print(f"{t:.6f}\t{ek:.10e}")