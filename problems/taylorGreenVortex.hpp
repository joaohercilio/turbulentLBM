#pragma once


#include <numbers>


#include "../core/definitions.hpp"
#include "../core/lbm.hpp"

constexpr dfloat U0 {0.009817477042468103};


static __global__ void setInitialConditions (dfloat* moments, int NX, int NY, int NZ, dfloat TAU)
{
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    const int x = bx * BLOCK_SIZE_X + tx;
    const int y = by * BLOCK_SIZE_Y + ty;
    const int z = bz * BLOCK_SIZE_Z + tz;

    if (x >= NX || y >= NY || z >= NZ) return;

    dfloat u0 = U0;
    dfloat k  = 2.0*std::numbers::pi/NX;
    dfloat ux =  u0 * sin(x*k) * cos(y*k) * cos(z*k);
    dfloat uy = -u0 * cos(x*k) * sin(y*k) * cos(z*k);
    dfloat drho = as2 * (u0*u0/16.0f) * (cos(2*k*x) + cos(2*k*y)) * (2 + cos(2*k*z));

    dfloat sxx = u0 * k * cos(k*x) * cos(k*y) * cos(k*z);
    dfloat syy = -sxx;
    dfloat sxz = -0.5 * u0 * k * sin(k*x) * cos(k*y) * sin(k*z);
    dfloat syz =  0.5 * u0 * k * cos(k*x) * sin(k*y) * sin(k*z);

    initMoments(moments, drho, ux, uy, 0, sxx, 0, sxz, syy, syz, 0, NX, NY, NZ, TAU);
}



/*
    TAYLOR GREEN VORTEX DECAY
    RE = 1600
    L     = N / (2 * pi)
    u0    = 1 / L
    nu    = 1 / RE
    tau = 0.5 + 3*nu
    N_STEPS = round(T/h**2)
*/
