#pragma once

#include "../core/definitions.hpp"

namespace d3q15_2nd {


constexpr int NUMBER_OF_VELOCITIES {15};

__device__ __constant__ constexpr int cx[15] = {  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1 };
__device__ __constant__ constexpr int cy[15] = {  0,  0,  0,  1, -1,  0,  0,  1, -1,  1, -1, -1,  1,  1, -1 };
__device__ __constant__ constexpr int cz[15] = {  0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1 };


static __device__ __forceinline__ dfloat reconstructPopulation(
    int k, dfloat drho, dfloat jx, dfloat jy, dfloat jz,
    dfloat pxx, dfloat pxy, dfloat pxz,
    dfloat pyy, dfloat pyz, dfloat pzz)
{
    switch (k) {

        case 0:
            return (2.0f/9.0f)*drho - 1.0f/3.0f*pxx - 1.0f/3.0f*pyy - 1.0f/3.0f*pzz;
        case 1:
            return (1.0f/9.0f)*drho + (1.0f/3.0f)*jx + (1.0f/3.0f)*pxx - 1.0f/6.0f*pyy - 1.0f/6.0f*pzz;
        case 2:
            return (1.0f/9.0f)*drho - 1.0f/3.0f*jx + (1.0f/3.0f)*pxx - 1.0f/6.0f*pyy - 1.0f/6.0f*pzz;
        case 3:
            return (1.0f/9.0f)*drho + (1.0f/3.0f)*jy - 1.0f/6.0f*pxx + (1.0f/3.0f)*pyy - 1.0f/6.0f*pzz;
        case 4:
            return (1.0f/9.0f)*drho - 1.0f/3.0f*jy - 1.0f/6.0f*pxx + (1.0f/3.0f)*pyy - 1.0f/6.0f*pzz;
        case 5:
            return (1.0f/9.0f)*drho + (1.0f/3.0f)*jz - 1.0f/6.0f*pxx - 1.0f/6.0f*pyy + (1.0f/3.0f)*pzz;
        case 6:
            return (1.0f/9.0f)*drho - 1.0f/3.0f*jz - 1.0f/6.0f*pxx - 1.0f/6.0f*pyy + (1.0f/3.0f)*pzz;
        case 7:
            return (1.0f/72.0f)*drho + (1.0f/24.0f)*jx + (1.0f/24.0f)*jy + (1.0f/24.0f)*jz + (1.0f/24.0f)*pxx + (1.0f/8.0f)*pxy + (1.0f/8.0f)*pxz + (1.0f/24.0f)*pyy + (1.0f/8.0f)*pyz + (1.0f/24.0f)*pzz;
        case 8:
            return (1.0f/72.0f)*drho - 1.0f/24.0f*jx - 1.0f/24.0f*jy - 1.0f/24.0f*jz + (1.0f/24.0f)*pxx + (1.0f/8.0f)*pxy + (1.0f/8.0f)*pxz + (1.0f/24.0f)*pyy + (1.0f/8.0f)*pyz + (1.0f/24.0f)*pzz;
        case 9:
            return (1.0f/72.0f)*drho + (1.0f/24.0f)*jx + (1.0f/24.0f)*jy - 1.0f/24.0f*jz + (1.0f/24.0f)*pxx + (1.0f/8.0f)*pxy - 1.0f/8.0f*pxz + (1.0f/24.0f)*pyy - 1.0f/8.0f*pyz + (1.0f/24.0f)*pzz;
        case 10:
            return (1.0f/72.0f)*drho - 1.0f/24.0f*jx - 1.0f/24.0f*jy + (1.0f/24.0f)*jz + (1.0f/24.0f)*pxx + (1.0f/8.0f)*pxy - 1.0f/8.0f*pxz + (1.0f/24.0f)*pyy - 1.0f/8.0f*pyz + (1.0f/24.0f)*pzz;
        case 11:
            return (1.0f/72.0f)*drho + (1.0f/24.0f)*jx - 1.0f/24.0f*jy + (1.0f/24.0f)*jz + (1.0f/24.0f)*pxx - 1.0f/8.0f*pxy + (1.0f/8.0f)*pxz + (1.0f/24.0f)*pyy - 1.0f/8.0f*pyz + (1.0f/24.0f)*pzz;
        case 12:
            return (1.0f/72.0f)*drho - 1.0f/24.0f*jx + (1.0f/24.0f)*jy - 1.0f/24.0f*jz + (1.0f/24.0f)*pxx - 1.0f/8.0f*pxy + (1.0f/8.0f)*pxz + (1.0f/24.0f)*pyy - 1.0f/8.0f*pyz + (1.0f/24.0f)*pzz;
        case 13:
            return (1.0f/72.0f)*drho - 1.0f/24.0f*jx + (1.0f/24.0f)*jy + (1.0f/24.0f)*jz + (1.0f/24.0f)*pxx - 1.0f/8.0f*pxy - 1.0f/8.0f*pxz + (1.0f/24.0f)*pyy + (1.0f/8.0f)*pyz + (1.0f/24.0f)*pzz;
        case 14:
            return (1.0f/72.0f)*drho + (1.0f/24.0f)*jx - 1.0f/24.0f*jy - 1.0f/24.0f*jz + (1.0f/24.0f)*pxx - 1.0f/8.0f*pxy - 1.0f/8.0f*pxz + (1.0f/24.0f)*pyy + (1.0f/8.0f)*pyz + (1.0f/24.0f)*pzz;
    }
    return 0;
}

}


namespace d3q15_3rd {


constexpr int NUMBER_OF_VELOCITIES {15};

__device__ __constant__ constexpr int cx[15] = {  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1 };
__device__ __constant__ constexpr int cy[15] = {  0,  0,  0,  1, -1,  0,  0,  1, -1,  1, -1, -1,  1,  1, -1 };
__device__ __constant__ constexpr int cz[15] = {  0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1 };


static __device__ __forceinline__ dfloat reconstructPopulation(
    int k, dfloat drho, dfloat jx, dfloat jy, dfloat jz,
    dfloat pxx, dfloat pxy, dfloat pxz,
    dfloat pyy, dfloat pyz, dfloat pzz)
{
    const dfloat rho = RHO_0 + drho;
    const dfloat ux = jx / rho, uy = jy / rho, uz = jz / rho;

    switch (k) {

        case 0:
            return (2.0f/9.0f)*drho - 1.0f/3.0f*pxx - 1.0f/3.0f*pyy - 1.0f/3.0f*pzz;
        case 1:
            return (1.0f/9.0f)*drho + (1.0f/3.0f)*jx + (1.0f/3.0f)*pxx - 1.0f/6.0f*pyy - 1.0f/6.0f*pzz - 1.0f/2.0f*uy*pxy - 1.0f/2.0f*uz*pxz - 1.0f/4.0f*ux*pyy - 1.0f/4.0f*ux*pzz + (1.0f/2.0f)*rho*ux*uy*uy + (1.0f/2.0f)*rho*ux*uz*uz;
        case 2:
            return (1.0f/9.0f)*drho - 1.0f/3.0f*jx + (1.0f/3.0f)*pxx - 1.0f/6.0f*pyy - 1.0f/6.0f*pzz + (1.0f/2.0f)*uy*pxy + (1.0f/2.0f)*uz*pxz + (1.0f/4.0f)*ux*pyy + (1.0f/4.0f)*ux*pzz - 1.0f/2.0f*rho*ux*uy*uy - 1.0f/2.0f*rho*ux*uz*uz;
        case 3:
            return (1.0f/9.0f)*drho + (1.0f/3.0f)*jy - 1.0f/6.0f*pxx + (1.0f/3.0f)*pyy - 1.0f/6.0f*pzz - 1.0f/4.0f*uy*pxx - 1.0f/2.0f*ux*pxy - 1.0f/2.0f*uz*pyz - 1.0f/4.0f*uy*pzz + (1.0f/2.0f)*rho*ux*ux*uy + (1.0f/2.0f)*rho*uy*uz*uz;
        case 4:
            return (1.0f/9.0f)*drho - 1.0f/3.0f*jy - 1.0f/6.0f*pxx + (1.0f/3.0f)*pyy - 1.0f/6.0f*pzz + (1.0f/4.0f)*uy*pxx + (1.0f/2.0f)*ux*pxy + (1.0f/2.0f)*uz*pyz + (1.0f/4.0f)*uy*pzz - 1.0f/2.0f*rho*ux*ux*uy - 1.0f/2.0f*rho*uy*uz*uz;
        case 5:
            return (1.0f/9.0f)*drho + (1.0f/3.0f)*jz - 1.0f/6.0f*pxx - 1.0f/6.0f*pyy + (1.0f/3.0f)*pzz - 1.0f/4.0f*uz*pxx - 1.0f/2.0f*ux*pxz - 1.0f/4.0f*uz*pyy - 1.0f/2.0f*uy*pyz + (1.0f/2.0f)*rho*ux*ux*uz + (1.0f/2.0f)*rho*uy*uy*uz;
        case 6:
            return (1.0f/9.0f)*drho - 1.0f/3.0f*jz - 1.0f/6.0f*pxx - 1.0f/6.0f*pyy + (1.0f/3.0f)*pzz + (1.0f/4.0f)*uz*pxx + (1.0f/2.0f)*ux*pxz + (1.0f/4.0f)*uz*pyy + (1.0f/2.0f)*uy*pyz - 1.0f/2.0f*rho*ux*ux*uz - 1.0f/2.0f*rho*uy*uy*uz;
        case 7:
            return (1.0f/72.0f)*drho + (1.0f/24.0f)*jx + (1.0f/24.0f)*jy + (1.0f/24.0f)*jz + (1.0f/24.0f)*pxx + (1.0f/8.0f)*pxy + (1.0f/8.0f)*pxz + (1.0f/24.0f)*pyy + (1.0f/8.0f)*pyz + (1.0f/24.0f)*pzz + (1.0f/16.0f)*uy*pxx + (1.0f/16.0f)*uz*pxx + (1.0f/8.0f)*ux*pxy + (1.0f/8.0f)*uy*pxy + (1.0f/8.0f)*uz*pxy + (1.0f/8.0f)*ux*pxz + (1.0f/8.0f)*uy*pxz + (1.0f/8.0f)*uz*pxz + (1.0f/16.0f)*ux*pyy + (1.0f/16.0f)*uz*pyy + (1.0f/8.0f)*ux*pyz + (1.0f/8.0f)*uy*pyz + (1.0f/8.0f)*uz*pyz + (1.0f/16.0f)*ux*pzz + (1.0f/16.0f)*uy*pzz - 1.0f/8.0f*rho*ux*ux*uy - 1.0f/8.0f*rho*ux*ux*uz - 1.0f/8.0f*rho*ux*uy*uy - 1.0f/4.0f*rho*ux*uy*uz - 1.0f/8.0f*rho*ux*uz*uz - 1.0f/8.0f*rho*uy*uy*uz - 1.0f/8.0f*rho*uy*uz*uz;
        case 8:
            return (1.0f/72.0f)*drho - 1.0f/24.0f*jx - 1.0f/24.0f*jy - 1.0f/24.0f*jz + (1.0f/24.0f)*pxx + (1.0f/8.0f)*pxy + (1.0f/8.0f)*pxz + (1.0f/24.0f)*pyy + (1.0f/8.0f)*pyz + (1.0f/24.0f)*pzz - 1.0f/16.0f*uy*pxx - 1.0f/16.0f*uz*pxx - 1.0f/8.0f*ux*pxy - 1.0f/8.0f*uy*pxy - 1.0f/8.0f*uz*pxy - 1.0f/8.0f*ux*pxz - 1.0f/8.0f*uy*pxz - 1.0f/8.0f*uz*pxz - 1.0f/16.0f*ux*pyy - 1.0f/16.0f*uz*pyy - 1.0f/8.0f*ux*pyz - 1.0f/8.0f*uy*pyz - 1.0f/8.0f*uz*pyz - 1.0f/16.0f*ux*pzz - 1.0f/16.0f*uy*pzz + (1.0f/8.0f)*rho*ux*ux*uy + (1.0f/8.0f)*rho*ux*ux*uz + (1.0f/8.0f)*rho*ux*uy*uy + (1.0f/4.0f)*rho*ux*uy*uz + (1.0f/8.0f)*rho*ux*uz*uz + (1.0f/8.0f)*rho*uy*uy*uz + (1.0f/8.0f)*rho*uy*uz*uz;
        case 9:
            return (1.0f/72.0f)*drho + (1.0f/24.0f)*jx + (1.0f/24.0f)*jy - 1.0f/24.0f*jz + (1.0f/24.0f)*pxx + (1.0f/8.0f)*pxy - 1.0f/8.0f*pxz + (1.0f/24.0f)*pyy - 1.0f/8.0f*pyz + (1.0f/24.0f)*pzz + (1.0f/16.0f)*uy*pxx - 1.0f/16.0f*uz*pxx + (1.0f/8.0f)*ux*pxy + (1.0f/8.0f)*uy*pxy - 1.0f/8.0f*uz*pxy - 1.0f/8.0f*ux*pxz - 1.0f/8.0f*uy*pxz + (1.0f/8.0f)*uz*pxz + (1.0f/16.0f)*ux*pyy - 1.0f/16.0f*uz*pyy - 1.0f/8.0f*ux*pyz - 1.0f/8.0f*uy*pyz + (1.0f/8.0f)*uz*pyz + (1.0f/16.0f)*ux*pzz + (1.0f/16.0f)*uy*pzz - 1.0f/8.0f*rho*ux*ux*uy + (1.0f/8.0f)*rho*ux*ux*uz - 1.0f/8.0f*rho*ux*uy*uy + (1.0f/4.0f)*rho*ux*uy*uz - 1.0f/8.0f*rho*ux*uz*uz + (1.0f/8.0f)*rho*uy*uy*uz - 1.0f/8.0f*rho*uy*uz*uz;
        case 10:
            return (1.0f/72.0f)*drho - 1.0f/24.0f*jx - 1.0f/24.0f*jy + (1.0f/24.0f)*jz + (1.0f/24.0f)*pxx + (1.0f/8.0f)*pxy - 1.0f/8.0f*pxz + (1.0f/24.0f)*pyy - 1.0f/8.0f*pyz + (1.0f/24.0f)*pzz - 1.0f/16.0f*uy*pxx + (1.0f/16.0f)*uz*pxx - 1.0f/8.0f*ux*pxy - 1.0f/8.0f*uy*pxy + (1.0f/8.0f)*uz*pxy + (1.0f/8.0f)*ux*pxz + (1.0f/8.0f)*uy*pxz - 1.0f/8.0f*uz*pxz - 1.0f/16.0f*ux*pyy + (1.0f/16.0f)*uz*pyy + (1.0f/8.0f)*ux*pyz + (1.0f/8.0f)*uy*pyz - 1.0f/8.0f*uz*pyz - 1.0f/16.0f*ux*pzz - 1.0f/16.0f*uy*pzz + (1.0f/8.0f)*rho*ux*ux*uy - 1.0f/8.0f*rho*ux*ux*uz + (1.0f/8.0f)*rho*ux*uy*uy - 1.0f/4.0f*rho*ux*uy*uz + (1.0f/8.0f)*rho*ux*uz*uz - 1.0f/8.0f*rho*uy*uy*uz + (1.0f/8.0f)*rho*uy*uz*uz;
        case 11:
            return (1.0f/72.0f)*drho + (1.0f/24.0f)*jx - 1.0f/24.0f*jy + (1.0f/24.0f)*jz + (1.0f/24.0f)*pxx - 1.0f/8.0f*pxy + (1.0f/8.0f)*pxz + (1.0f/24.0f)*pyy - 1.0f/8.0f*pyz + (1.0f/24.0f)*pzz - 1.0f/16.0f*uy*pxx + (1.0f/16.0f)*uz*pxx - 1.0f/8.0f*ux*pxy + (1.0f/8.0f)*uy*pxy - 1.0f/8.0f*uz*pxy + (1.0f/8.0f)*ux*pxz - 1.0f/8.0f*uy*pxz + (1.0f/8.0f)*uz*pxz + (1.0f/16.0f)*ux*pyy + (1.0f/16.0f)*uz*pyy - 1.0f/8.0f*ux*pyz + (1.0f/8.0f)*uy*pyz - 1.0f/8.0f*uz*pyz + (1.0f/16.0f)*ux*pzz - 1.0f/16.0f*uy*pzz + (1.0f/8.0f)*rho*ux*ux*uy - 1.0f/8.0f*rho*ux*ux*uz - 1.0f/8.0f*rho*ux*uy*uy + (1.0f/4.0f)*rho*ux*uy*uz - 1.0f/8.0f*rho*ux*uz*uz - 1.0f/8.0f*rho*uy*uy*uz + (1.0f/8.0f)*rho*uy*uz*uz;
        case 12:
            return (1.0f/72.0f)*drho - 1.0f/24.0f*jx + (1.0f/24.0f)*jy - 1.0f/24.0f*jz + (1.0f/24.0f)*pxx - 1.0f/8.0f*pxy + (1.0f/8.0f)*pxz + (1.0f/24.0f)*pyy - 1.0f/8.0f*pyz + (1.0f/24.0f)*pzz + (1.0f/16.0f)*uy*pxx - 1.0f/16.0f*uz*pxx + (1.0f/8.0f)*ux*pxy - 1.0f/8.0f*uy*pxy + (1.0f/8.0f)*uz*pxy - 1.0f/8.0f*ux*pxz + (1.0f/8.0f)*uy*pxz - 1.0f/8.0f*uz*pxz - 1.0f/16.0f*ux*pyy - 1.0f/16.0f*uz*pyy + (1.0f/8.0f)*ux*pyz - 1.0f/8.0f*uy*pyz + (1.0f/8.0f)*uz*pyz - 1.0f/16.0f*ux*pzz + (1.0f/16.0f)*uy*pzz - 1.0f/8.0f*rho*ux*ux*uy + (1.0f/8.0f)*rho*ux*ux*uz + (1.0f/8.0f)*rho*ux*uy*uy - 1.0f/4.0f*rho*ux*uy*uz + (1.0f/8.0f)*rho*ux*uz*uz + (1.0f/8.0f)*rho*uy*uy*uz - 1.0f/8.0f*rho*uy*uz*uz;
        case 13:
            return (1.0f/72.0f)*drho - 1.0f/24.0f*jx + (1.0f/24.0f)*jy + (1.0f/24.0f)*jz + (1.0f/24.0f)*pxx - 1.0f/8.0f*pxy - 1.0f/8.0f*pxz + (1.0f/24.0f)*pyy + (1.0f/8.0f)*pyz + (1.0f/24.0f)*pzz + (1.0f/16.0f)*uy*pxx + (1.0f/16.0f)*uz*pxx + (1.0f/8.0f)*ux*pxy - 1.0f/8.0f*uy*pxy - 1.0f/8.0f*uz*pxy + (1.0f/8.0f)*ux*pxz - 1.0f/8.0f*uy*pxz - 1.0f/8.0f*uz*pxz - 1.0f/16.0f*ux*pyy + (1.0f/16.0f)*uz*pyy - 1.0f/8.0f*ux*pyz + (1.0f/8.0f)*uy*pyz + (1.0f/8.0f)*uz*pyz - 1.0f/16.0f*ux*pzz + (1.0f/16.0f)*uy*pzz - 1.0f/8.0f*rho*ux*ux*uy - 1.0f/8.0f*rho*ux*ux*uz + (1.0f/8.0f)*rho*ux*uy*uy + (1.0f/4.0f)*rho*ux*uy*uz + (1.0f/8.0f)*rho*ux*uz*uz - 1.0f/8.0f*rho*uy*uy*uz - 1.0f/8.0f*rho*uy*uz*uz;
        case 14:
            return (1.0f/72.0f)*drho + (1.0f/24.0f)*jx - 1.0f/24.0f*jy - 1.0f/24.0f*jz + (1.0f/24.0f)*pxx - 1.0f/8.0f*pxy - 1.0f/8.0f*pxz + (1.0f/24.0f)*pyy + (1.0f/8.0f)*pyz + (1.0f/24.0f)*pzz - 1.0f/16.0f*uy*pxx - 1.0f/16.0f*uz*pxx - 1.0f/8.0f*ux*pxy + (1.0f/8.0f)*uy*pxy + (1.0f/8.0f)*uz*pxy - 1.0f/8.0f*ux*pxz + (1.0f/8.0f)*uy*pxz + (1.0f/8.0f)*uz*pxz + (1.0f/16.0f)*ux*pyy - 1.0f/16.0f*uz*pyy + (1.0f/8.0f)*ux*pyz - 1.0f/8.0f*uy*pyz - 1.0f/8.0f*uz*pyz + (1.0f/16.0f)*ux*pzz - 1.0f/16.0f*uy*pzz + (1.0f/8.0f)*rho*ux*ux*uy + (1.0f/8.0f)*rho*ux*ux*uz - 1.0f/8.0f*rho*ux*uy*uy - 1.0f/4.0f*rho*ux*uy*uz - 1.0f/8.0f*rho*ux*uz*uz + (1.0f/8.0f)*rho*uy*uy*uz + (1.0f/8.0f)*rho*uy*uz*uz;
    }
    return 0;
}

}
