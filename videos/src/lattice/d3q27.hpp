#pragma once

#include "../core/definitions.hpp"

namespace d3q27_2nd {


constexpr int NUMBER_OF_VELOCITIES {27};

__device__ __constant__ constexpr int cx[27] = { 0,  1, -1,  0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1,  0,  0,  0,  0, -1,  1,  1, -1,  1, -1,  1, -1};
__device__ __constant__ constexpr int cy[27] = { 0,  0,  0,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1};
__device__ __constant__ constexpr int cz[27] = { 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1};


static __device__ __forceinline__ dfloat reconstructPopulation(
    int k, dfloat drho, dfloat jx, dfloat jy, dfloat jz,
    dfloat pxx, dfloat pxy, dfloat pxz,
    dfloat pyy, dfloat pyz, dfloat pzz)
{
    switch (k) {

        case 0:
            return -4.0f/9.0f*pxx - 4.0f/9.0f*pyy - 4.0f/9.0f*pzz + (8.0f/27.0f)*drho;
        case 1:
            return (2.0f/9.0f)*pxx - 1.0f/9.0f*pyy - 1.0f/9.0f*pzz + (2.0f/27.0f)*drho + (2.0f/9.0f)*jx;
        case 2:
            return (2.0f/9.0f)*pxx - 1.0f/9.0f*pyy - 1.0f/9.0f*pzz + (2.0f/27.0f)*drho - 2.0f/9.0f*jx;
        case 3:
            return -1.0f/9.0f*pxx + (2.0f/9.0f)*pyy - 1.0f/9.0f*pzz + (2.0f/27.0f)*drho + (2.0f/9.0f)*jy;
        case 4:
            return -1.0f/9.0f*pxx + (2.0f/9.0f)*pyy - 1.0f/9.0f*pzz + (2.0f/27.0f)*drho - 2.0f/9.0f*jy;
        case 5:
            return -1.0f/9.0f*pxx - 1.0f/9.0f*pyy + (2.0f/9.0f)*pzz + (2.0f/27.0f)*drho + (2.0f/9.0f)*jz;
        case 6:
            return -1.0f/9.0f*pxx - 1.0f/9.0f*pyy + (2.0f/9.0f)*pzz + (2.0f/27.0f)*drho - 2.0f/9.0f*jz;
        case 7:
            return (1.0f/18.0f)*pxx + (1.0f/6.0f)*pxy + (1.0f/18.0f)*pyy - 1.0f/36.0f*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jx + (1.0f/18.0f)*jy;
        case 8:
            return (1.0f/18.0f)*pxx + (1.0f/6.0f)*pxy + (1.0f/18.0f)*pyy - 1.0f/36.0f*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jx - 1.0f/18.0f*jy;
        case 9:
            return (1.0f/18.0f)*pxx - 1.0f/6.0f*pxy + (1.0f/18.0f)*pyy - 1.0f/36.0f*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jx + (1.0f/18.0f)*jy;
        case 10:
            return (1.0f/18.0f)*pxx - 1.0f/6.0f*pxy + (1.0f/18.0f)*pyy - 1.0f/36.0f*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jx - 1.0f/18.0f*jy;
        case 11:
            return (1.0f/18.0f)*pxx + (1.0f/6.0f)*pxz - 1.0f/36.0f*pyy + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jx + (1.0f/18.0f)*jz;
        case 12:
            return (1.0f/18.0f)*pxx + (1.0f/6.0f)*pxz - 1.0f/36.0f*pyy + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jx - 1.0f/18.0f*jz;
        case 13:
            return (1.0f/18.0f)*pxx - 1.0f/6.0f*pxz - 1.0f/36.0f*pyy + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jx + (1.0f/18.0f)*jz;
        case 14:
            return (1.0f/18.0f)*pxx - 1.0f/6.0f*pxz - 1.0f/36.0f*pyy + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jx - 1.0f/18.0f*jz;
        case 15:
            return -1.0f/36.0f*pxx + (1.0f/18.0f)*pyy + (1.0f/6.0f)*pyz + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jy + (1.0f/18.0f)*jz;
        case 16:
            return -1.0f/36.0f*pxx + (1.0f/18.0f)*pyy + (1.0f/6.0f)*pyz + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jy - 1.0f/18.0f*jz;
        case 17:
            return -1.0f/36.0f*pxx + (1.0f/18.0f)*pyy - 1.0f/6.0f*pyz + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jy - 1.0f/18.0f*jz;
        case 18:
            return -1.0f/36.0f*pxx + (1.0f/18.0f)*pyy - 1.0f/6.0f*pyz + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jy + (1.0f/18.0f)*jz;
        case 19:
            return (1.0f/72.0f)*pxx - 1.0f/24.0f*pxy - 1.0f/24.0f*pxz + (1.0f/72.0f)*pyy + (1.0f/24.0f)*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho - 1.0f/72.0f*jx + (1.0f/72.0f)*jy + (1.0f/72.0f)*jz;
        case 20:
            return (1.0f/72.0f)*pxx - 1.0f/24.0f*pxy - 1.0f/24.0f*pxz + (1.0f/72.0f)*pyy + (1.0f/24.0f)*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho + (1.0f/72.0f)*jx - 1.0f/72.0f*jy - 1.0f/72.0f*jz;
        case 21:
            return (1.0f/72.0f)*pxx - 1.0f/24.0f*pxy + (1.0f/24.0f)*pxz + (1.0f/72.0f)*pyy - 1.0f/24.0f*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho + (1.0f/72.0f)*jx - 1.0f/72.0f*jy + (1.0f/72.0f)*jz;
        case 22:
            return (1.0f/72.0f)*pxx - 1.0f/24.0f*pxy + (1.0f/24.0f)*pxz + (1.0f/72.0f)*pyy - 1.0f/24.0f*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho - 1.0f/72.0f*jx + (1.0f/72.0f)*jy - 1.0f/72.0f*jz;
        case 23:
            return (1.0f/72.0f)*pxx + (1.0f/24.0f)*pxy - 1.0f/24.0f*pxz + (1.0f/72.0f)*pyy - 1.0f/24.0f*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho + (1.0f/72.0f)*jx + (1.0f/72.0f)*jy - 1.0f/72.0f*jz;
        case 24:
            return (1.0f/72.0f)*pxx + (1.0f/24.0f)*pxy - 1.0f/24.0f*pxz + (1.0f/72.0f)*pyy - 1.0f/24.0f*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho - 1.0f/72.0f*jx - 1.0f/72.0f*jy + (1.0f/72.0f)*jz;
        case 25:
            return (1.0f/72.0f)*pxx + (1.0f/24.0f)*pxy + (1.0f/24.0f)*pxz + (1.0f/72.0f)*pyy + (1.0f/24.0f)*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho + (1.0f/72.0f)*jx + (1.0f/72.0f)*jy + (1.0f/72.0f)*jz;
        case 26:
            return (1.0f/72.0f)*pxx + (1.0f/24.0f)*pxy + (1.0f/24.0f)*pxz + (1.0f/72.0f)*pyy + (1.0f/24.0f)*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho - 1.0f/72.0f*jx - 1.0f/72.0f*jy - 1.0f/72.0f*jz;
    }
    return 0;
}

}


namespace d3q27_3rd {


constexpr int NUMBER_OF_VELOCITIES {27};

__device__ __constant__ constexpr int cx[27] = { 0,  1, -1,  0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1,  0,  0,  0,  0, -1,  1,  1, -1,  1, -1,  1, -1};
__device__ __constant__ constexpr int cy[27] = { 0,  0,  0,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1};
__device__ __constant__ constexpr int cz[27] = { 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1};


static __device__ __forceinline__ dfloat reconstructPopulation(
    int k, dfloat drho, dfloat jx, dfloat jy, dfloat jz,
    dfloat pxx, dfloat pxy, dfloat pxz,
    dfloat pyy, dfloat pyz, dfloat pzz)
{
    const dfloat rho = RHO_0 + drho;
    const dfloat ux = jx / rho, uy = jy / rho, uz = jz / rho;

    switch (k) {

        case 0:
            return -4.0f/9.0f*pxx - 4.0f/9.0f*pyy - 4.0f/9.0f*pzz + (8.0f/27.0f)*drho;
        case 1:
            return (2.0f/9.0f)*pxx - 1.0f/9.0f*pyy - 1.0f/9.0f*pzz + (2.0f/27.0f)*drho + (2.0f/9.0f)*jx - 2.0f/3.0f*uy*pxy - 2.0f/3.0f*uz*pxz - 1.0f/3.0f*ux*pyy - 1.0f/3.0f*ux*pzz + (2.0f/3.0f)*rho*ux*uy*uy + (2.0f/3.0f)*rho*ux*uz*uz;
        case 2:
            return (2.0f/9.0f)*pxx - 1.0f/9.0f*pyy - 1.0f/9.0f*pzz + (2.0f/27.0f)*drho - 2.0f/9.0f*jx + (2.0f/3.0f)*uy*pxy + (2.0f/3.0f)*uz*pxz + (1.0f/3.0f)*ux*pyy + (1.0f/3.0f)*ux*pzz - 2.0f/3.0f*rho*ux*uy*uy - 2.0f/3.0f*rho*ux*uz*uz;
        case 3:
            return -1.0f/9.0f*pxx + (2.0f/9.0f)*pyy - 1.0f/9.0f*pzz + (2.0f/27.0f)*drho + (2.0f/9.0f)*jy - 1.0f/3.0f*uy*pxx - 2.0f/3.0f*ux*pxy - 2.0f/3.0f*uz*pyz - 1.0f/3.0f*uy*pzz + (2.0f/3.0f)*rho*ux*ux*uy + (2.0f/3.0f)*rho*uy*uz*uz;
        case 4:
            return -1.0f/9.0f*pxx + (2.0f/9.0f)*pyy - 1.0f/9.0f*pzz + (2.0f/27.0f)*drho - 2.0f/9.0f*jy + (1.0f/3.0f)*uy*pxx + (2.0f/3.0f)*ux*pxy + (2.0f/3.0f)*uz*pyz + (1.0f/3.0f)*uy*pzz - 2.0f/3.0f*rho*ux*ux*uy - 2.0f/3.0f*rho*uy*uz*uz;
        case 5:
            return -1.0f/9.0f*pxx - 1.0f/9.0f*pyy + (2.0f/9.0f)*pzz + (2.0f/27.0f)*drho + (2.0f/9.0f)*jz - 1.0f/3.0f*uz*pxx - 2.0f/3.0f*ux*pxz - 1.0f/3.0f*uz*pyy - 2.0f/3.0f*uy*pyz + (2.0f/3.0f)*rho*ux*ux*uz + (2.0f/3.0f)*rho*uy*uy*uz;
        case 6:
            return -1.0f/9.0f*pxx - 1.0f/9.0f*pyy + (2.0f/9.0f)*pzz + (2.0f/27.0f)*drho - 2.0f/9.0f*jz + (1.0f/3.0f)*uz*pxx + (2.0f/3.0f)*ux*pxz + (1.0f/3.0f)*uz*pyy + (2.0f/3.0f)*uy*pyz - 2.0f/3.0f*rho*ux*ux*uz - 2.0f/3.0f*rho*uy*uy*uz;
        case 7:
            return (1.0f/18.0f)*pxx + (1.0f/6.0f)*pxy + (1.0f/18.0f)*pyy - 1.0f/36.0f*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jx + (1.0f/18.0f)*jy + (1.0f/6.0f)*uy*pxx + (1.0f/3.0f)*ux*pxy + (1.0f/3.0f)*uy*pxy - 1.0f/6.0f*uz*pxz + (1.0f/6.0f)*ux*pyy - 1.0f/6.0f*uz*pyz - 1.0f/12.0f*ux*pzz - 1.0f/12.0f*uy*pzz - 1.0f/3.0f*rho*ux*ux*uy - 1.0f/3.0f*rho*ux*uy*uy + (1.0f/6.0f)*rho*ux*uz*uz + (1.0f/6.0f)*rho*uy*uz*uz;
        case 8:
            return (1.0f/18.0f)*pxx + (1.0f/6.0f)*pxy + (1.0f/18.0f)*pyy - 1.0f/36.0f*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jx - 1.0f/18.0f*jy - 1.0f/6.0f*uy*pxx - 1.0f/3.0f*ux*pxy - 1.0f/3.0f*uy*pxy + (1.0f/6.0f)*uz*pxz - 1.0f/6.0f*ux*pyy + (1.0f/6.0f)*uz*pyz + (1.0f/12.0f)*ux*pzz + (1.0f/12.0f)*uy*pzz + (1.0f/3.0f)*rho*ux*ux*uy + (1.0f/3.0f)*rho*ux*uy*uy - 1.0f/6.0f*rho*ux*uz*uz - 1.0f/6.0f*rho*uy*uz*uz;
        case 9:
            return (1.0f/18.0f)*pxx - 1.0f/6.0f*pxy + (1.0f/18.0f)*pyy - 1.0f/36.0f*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jx + (1.0f/18.0f)*jy + (1.0f/6.0f)*uy*pxx + (1.0f/3.0f)*ux*pxy - 1.0f/3.0f*uy*pxy + (1.0f/6.0f)*uz*pxz - 1.0f/6.0f*ux*pyy - 1.0f/6.0f*uz*pyz + (1.0f/12.0f)*ux*pzz - 1.0f/12.0f*uy*pzz - 1.0f/3.0f*rho*ux*ux*uy + (1.0f/3.0f)*rho*ux*uy*uy - 1.0f/6.0f*rho*ux*uz*uz + (1.0f/6.0f)*rho*uy*uz*uz;
        case 10:
            return (1.0f/18.0f)*pxx - 1.0f/6.0f*pxy + (1.0f/18.0f)*pyy - 1.0f/36.0f*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jx - 1.0f/18.0f*jy - 1.0f/6.0f*uy*pxx - 1.0f/3.0f*ux*pxy + (1.0f/3.0f)*uy*pxy - 1.0f/6.0f*uz*pxz + (1.0f/6.0f)*ux*pyy + (1.0f/6.0f)*uz*pyz - 1.0f/12.0f*ux*pzz + (1.0f/12.0f)*uy*pzz + (1.0f/3.0f)*rho*ux*ux*uy - 1.0f/3.0f*rho*ux*uy*uy + (1.0f/6.0f)*rho*ux*uz*uz - 1.0f/6.0f*rho*uy*uz*uz;
        case 11:
            return (1.0f/18.0f)*pxx + (1.0f/6.0f)*pxz - 1.0f/36.0f*pyy + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jx + (1.0f/18.0f)*jz + (1.0f/6.0f)*uz*pxx - 1.0f/6.0f*uy*pxy + (1.0f/3.0f)*ux*pxz + (1.0f/3.0f)*uz*pxz - 1.0f/12.0f*ux*pyy - 1.0f/12.0f*uz*pyy - 1.0f/6.0f*uy*pyz + (1.0f/6.0f)*ux*pzz - 1.0f/3.0f*rho*ux*ux*uz + (1.0f/6.0f)*rho*ux*uy*uy - 1.0f/3.0f*rho*ux*uz*uz + (1.0f/6.0f)*rho*uy*uy*uz;
        case 12:
            return (1.0f/18.0f)*pxx + (1.0f/6.0f)*pxz - 1.0f/36.0f*pyy + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jx - 1.0f/18.0f*jz - 1.0f/6.0f*uz*pxx + (1.0f/6.0f)*uy*pxy - 1.0f/3.0f*ux*pxz - 1.0f/3.0f*uz*pxz + (1.0f/12.0f)*ux*pyy + (1.0f/12.0f)*uz*pyy + (1.0f/6.0f)*uy*pyz - 1.0f/6.0f*ux*pzz + (1.0f/3.0f)*rho*ux*ux*uz - 1.0f/6.0f*rho*ux*uy*uy + (1.0f/3.0f)*rho*ux*uz*uz - 1.0f/6.0f*rho*uy*uy*uz;
        case 13:
            return (1.0f/18.0f)*pxx - 1.0f/6.0f*pxz - 1.0f/36.0f*pyy + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jx + (1.0f/18.0f)*jz + (1.0f/6.0f)*uz*pxx + (1.0f/6.0f)*uy*pxy + (1.0f/3.0f)*ux*pxz - 1.0f/3.0f*uz*pxz + (1.0f/12.0f)*ux*pyy - 1.0f/12.0f*uz*pyy - 1.0f/6.0f*uy*pyz - 1.0f/6.0f*ux*pzz - 1.0f/3.0f*rho*ux*ux*uz - 1.0f/6.0f*rho*ux*uy*uy + (1.0f/3.0f)*rho*ux*uz*uz + (1.0f/6.0f)*rho*uy*uy*uz;
        case 14:
            return (1.0f/18.0f)*pxx - 1.0f/6.0f*pxz - 1.0f/36.0f*pyy + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jx - 1.0f/18.0f*jz - 1.0f/6.0f*uz*pxx - 1.0f/6.0f*uy*pxy - 1.0f/3.0f*ux*pxz + (1.0f/3.0f)*uz*pxz - 1.0f/12.0f*ux*pyy + (1.0f/12.0f)*uz*pyy + (1.0f/6.0f)*uy*pyz + (1.0f/6.0f)*ux*pzz + (1.0f/3.0f)*rho*ux*ux*uz + (1.0f/6.0f)*rho*ux*uy*uy - 1.0f/3.0f*rho*ux*uz*uz - 1.0f/6.0f*rho*uy*uy*uz;
        case 15:
            return -1.0f/36.0f*pxx + (1.0f/18.0f)*pyy + (1.0f/6.0f)*pyz + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jy + (1.0f/18.0f)*jz - 1.0f/12.0f*uy*pxx - 1.0f/12.0f*uz*pxx - 1.0f/6.0f*ux*pxy - 1.0f/6.0f*ux*pxz + (1.0f/6.0f)*uz*pyy + (1.0f/3.0f)*uy*pyz + (1.0f/3.0f)*uz*pyz + (1.0f/6.0f)*uy*pzz + (1.0f/6.0f)*rho*ux*ux*uy + (1.0f/6.0f)*rho*ux*ux*uz - 1.0f/3.0f*rho*uy*uy*uz - 1.0f/3.0f*rho*uy*uz*uz;
        case 16:
            return -1.0f/36.0f*pxx + (1.0f/18.0f)*pyy + (1.0f/6.0f)*pyz + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jy - 1.0f/18.0f*jz + (1.0f/12.0f)*uy*pxx + (1.0f/12.0f)*uz*pxx + (1.0f/6.0f)*ux*pxy + (1.0f/6.0f)*ux*pxz - 1.0f/6.0f*uz*pyy - 1.0f/3.0f*uy*pyz - 1.0f/3.0f*uz*pyz - 1.0f/6.0f*uy*pzz - 1.0f/6.0f*rho*ux*ux*uy - 1.0f/6.0f*rho*ux*ux*uz + (1.0f/3.0f)*rho*uy*uy*uz + (1.0f/3.0f)*rho*uy*uz*uz;
        case 17:
            return -1.0f/36.0f*pxx + (1.0f/18.0f)*pyy - 1.0f/6.0f*pyz + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho + (1.0f/18.0f)*jy - 1.0f/18.0f*jz - 1.0f/12.0f*uy*pxx + (1.0f/12.0f)*uz*pxx - 1.0f/6.0f*ux*pxy + (1.0f/6.0f)*ux*pxz - 1.0f/6.0f*uz*pyy - 1.0f/3.0f*uy*pyz + (1.0f/3.0f)*uz*pyz + (1.0f/6.0f)*uy*pzz + (1.0f/6.0f)*rho*ux*ux*uy - 1.0f/6.0f*rho*ux*ux*uz + (1.0f/3.0f)*rho*uy*uy*uz - 1.0f/3.0f*rho*uy*uz*uz;
        case 18:
            return -1.0f/36.0f*pxx + (1.0f/18.0f)*pyy - 1.0f/6.0f*pyz + (1.0f/18.0f)*pzz + (1.0f/54.0f)*drho - 1.0f/18.0f*jy + (1.0f/18.0f)*jz + (1.0f/12.0f)*uy*pxx - 1.0f/12.0f*uz*pxx + (1.0f/6.0f)*ux*pxy - 1.0f/6.0f*ux*pxz + (1.0f/6.0f)*uz*pyy + (1.0f/3.0f)*uy*pyz - 1.0f/3.0f*uz*pyz - 1.0f/6.0f*uy*pzz - 1.0f/6.0f*rho*ux*ux*uy + (1.0f/6.0f)*rho*ux*ux*uz - 1.0f/3.0f*rho*uy*uy*uz + (1.0f/3.0f)*rho*uy*uz*uz;
        case 19:
            return (1.0f/72.0f)*pxx - 1.0f/24.0f*pxy - 1.0f/24.0f*pxz + (1.0f/72.0f)*pyy + (1.0f/24.0f)*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho - 1.0f/72.0f*jx + (1.0f/72.0f)*jy + (1.0f/72.0f)*jz + (1.0f/24.0f)*uy*pxx + (1.0f/24.0f)*uz*pxx + (1.0f/12.0f)*ux*pxy - 1.0f/12.0f*uy*pxy - 1.0f/8.0f*uz*pxy + (1.0f/12.0f)*ux*pxz - 1.0f/8.0f*uy*pxz - 1.0f/12.0f*uz*pxz - 1.0f/24.0f*ux*pyy + (1.0f/24.0f)*uz*pyy - 1.0f/8.0f*ux*pyz + (1.0f/12.0f)*uy*pyz + (1.0f/12.0f)*uz*pyz - 1.0f/24.0f*ux*pzz + (1.0f/24.0f)*uy*pzz - 1.0f/12.0f*rho*ux*ux*uy - 1.0f/12.0f*rho*ux*ux*uz + (1.0f/12.0f)*rho*ux*uy*uy + (1.0f/4.0f)*rho*ux*uy*uz + (1.0f/12.0f)*rho*ux*uz*uz - 1.0f/12.0f*rho*uy*uy*uz - 1.0f/12.0f*rho*uy*uz*uz;
        case 20:
            return (1.0f/72.0f)*pxx - 1.0f/24.0f*pxy - 1.0f/24.0f*pxz + (1.0f/72.0f)*pyy + (1.0f/24.0f)*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho + (1.0f/72.0f)*jx - 1.0f/72.0f*jy - 1.0f/72.0f*jz - 1.0f/24.0f*uy*pxx - 1.0f/24.0f*uz*pxx - 1.0f/12.0f*ux*pxy + (1.0f/12.0f)*uy*pxy + (1.0f/8.0f)*uz*pxy - 1.0f/12.0f*ux*pxz + (1.0f/8.0f)*uy*pxz + (1.0f/12.0f)*uz*pxz + (1.0f/24.0f)*ux*pyy - 1.0f/24.0f*uz*pyy + (1.0f/8.0f)*ux*pyz - 1.0f/12.0f*uy*pyz - 1.0f/12.0f*uz*pyz + (1.0f/24.0f)*ux*pzz - 1.0f/24.0f*uy*pzz + (1.0f/12.0f)*rho*ux*ux*uy + (1.0f/12.0f)*rho*ux*ux*uz - 1.0f/12.0f*rho*ux*uy*uy - 1.0f/4.0f*rho*ux*uy*uz - 1.0f/12.0f*rho*ux*uz*uz + (1.0f/12.0f)*rho*uy*uy*uz + (1.0f/12.0f)*rho*uy*uz*uz;
        case 21:
            return (1.0f/72.0f)*pxx - 1.0f/24.0f*pxy + (1.0f/24.0f)*pxz + (1.0f/72.0f)*pyy - 1.0f/24.0f*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho + (1.0f/72.0f)*jx - 1.0f/72.0f*jy + (1.0f/72.0f)*jz - 1.0f/24.0f*uy*pxx + (1.0f/24.0f)*uz*pxx - 1.0f/12.0f*ux*pxy + (1.0f/12.0f)*uy*pxy - 1.0f/8.0f*uz*pxy + (1.0f/12.0f)*ux*pxz - 1.0f/8.0f*uy*pxz + (1.0f/12.0f)*uz*pxz + (1.0f/24.0f)*ux*pyy + (1.0f/24.0f)*uz*pyy - 1.0f/8.0f*ux*pyz + (1.0f/12.0f)*uy*pyz - 1.0f/12.0f*uz*pyz + (1.0f/24.0f)*ux*pzz - 1.0f/24.0f*uy*pzz + (1.0f/12.0f)*rho*ux*ux*uy - 1.0f/12.0f*rho*ux*ux*uz - 1.0f/12.0f*rho*ux*uy*uy + (1.0f/4.0f)*rho*ux*uy*uz - 1.0f/12.0f*rho*ux*uz*uz - 1.0f/12.0f*rho*uy*uy*uz + (1.0f/12.0f)*rho*uy*uz*uz;
        case 22:
            return (1.0f/72.0f)*pxx - 1.0f/24.0f*pxy + (1.0f/24.0f)*pxz + (1.0f/72.0f)*pyy - 1.0f/24.0f*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho - 1.0f/72.0f*jx + (1.0f/72.0f)*jy - 1.0f/72.0f*jz + (1.0f/24.0f)*uy*pxx - 1.0f/24.0f*uz*pxx + (1.0f/12.0f)*ux*pxy - 1.0f/12.0f*uy*pxy + (1.0f/8.0f)*uz*pxy - 1.0f/12.0f*ux*pxz + (1.0f/8.0f)*uy*pxz - 1.0f/12.0f*uz*pxz - 1.0f/24.0f*ux*pyy - 1.0f/24.0f*uz*pyy + (1.0f/8.0f)*ux*pyz - 1.0f/12.0f*uy*pyz + (1.0f/12.0f)*uz*pyz - 1.0f/24.0f*ux*pzz + (1.0f/24.0f)*uy*pzz - 1.0f/12.0f*rho*ux*ux*uy + (1.0f/12.0f)*rho*ux*ux*uz + (1.0f/12.0f)*rho*ux*uy*uy - 1.0f/4.0f*rho*ux*uy*uz + (1.0f/12.0f)*rho*ux*uz*uz + (1.0f/12.0f)*rho*uy*uy*uz - 1.0f/12.0f*rho*uy*uz*uz;
        case 23:
            return (1.0f/72.0f)*pxx + (1.0f/24.0f)*pxy - 1.0f/24.0f*pxz + (1.0f/72.0f)*pyy - 1.0f/24.0f*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho + (1.0f/72.0f)*jx + (1.0f/72.0f)*jy - 1.0f/72.0f*jz + (1.0f/24.0f)*uy*pxx - 1.0f/24.0f*uz*pxx + (1.0f/12.0f)*ux*pxy + (1.0f/12.0f)*uy*pxy - 1.0f/8.0f*uz*pxy - 1.0f/12.0f*ux*pxz - 1.0f/8.0f*uy*pxz + (1.0f/12.0f)*uz*pxz + (1.0f/24.0f)*ux*pyy - 1.0f/24.0f*uz*pyy - 1.0f/8.0f*ux*pyz - 1.0f/12.0f*uy*pyz + (1.0f/12.0f)*uz*pyz + (1.0f/24.0f)*ux*pzz + (1.0f/24.0f)*uy*pzz - 1.0f/12.0f*rho*ux*ux*uy + (1.0f/12.0f)*rho*ux*ux*uz - 1.0f/12.0f*rho*ux*uy*uy + (1.0f/4.0f)*rho*ux*uy*uz - 1.0f/12.0f*rho*ux*uz*uz + (1.0f/12.0f)*rho*uy*uy*uz - 1.0f/12.0f*rho*uy*uz*uz;
        case 24:
            return (1.0f/72.0f)*pxx + (1.0f/24.0f)*pxy - 1.0f/24.0f*pxz + (1.0f/72.0f)*pyy - 1.0f/24.0f*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho - 1.0f/72.0f*jx - 1.0f/72.0f*jy + (1.0f/72.0f)*jz - 1.0f/24.0f*uy*pxx + (1.0f/24.0f)*uz*pxx - 1.0f/12.0f*ux*pxy - 1.0f/12.0f*uy*pxy + (1.0f/8.0f)*uz*pxy + (1.0f/12.0f)*ux*pxz + (1.0f/8.0f)*uy*pxz - 1.0f/12.0f*uz*pxz - 1.0f/24.0f*ux*pyy + (1.0f/24.0f)*uz*pyy + (1.0f/8.0f)*ux*pyz + (1.0f/12.0f)*uy*pyz - 1.0f/12.0f*uz*pyz - 1.0f/24.0f*ux*pzz - 1.0f/24.0f*uy*pzz + (1.0f/12.0f)*rho*ux*ux*uy - 1.0f/12.0f*rho*ux*ux*uz + (1.0f/12.0f)*rho*ux*uy*uy - 1.0f/4.0f*rho*ux*uy*uz + (1.0f/12.0f)*rho*ux*uz*uz - 1.0f/12.0f*rho*uy*uy*uz + (1.0f/12.0f)*rho*uy*uz*uz;
        case 25:
            return (1.0f/72.0f)*pxx + (1.0f/24.0f)*pxy + (1.0f/24.0f)*pxz + (1.0f/72.0f)*pyy + (1.0f/24.0f)*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho + (1.0f/72.0f)*jx + (1.0f/72.0f)*jy + (1.0f/72.0f)*jz + (1.0f/24.0f)*uy*pxx + (1.0f/24.0f)*uz*pxx + (1.0f/12.0f)*ux*pxy + (1.0f/12.0f)*uy*pxy + (1.0f/8.0f)*uz*pxy + (1.0f/12.0f)*ux*pxz + (1.0f/8.0f)*uy*pxz + (1.0f/12.0f)*uz*pxz + (1.0f/24.0f)*ux*pyy + (1.0f/24.0f)*uz*pyy + (1.0f/8.0f)*ux*pyz + (1.0f/12.0f)*uy*pyz + (1.0f/12.0f)*uz*pyz + (1.0f/24.0f)*ux*pzz + (1.0f/24.0f)*uy*pzz - 1.0f/12.0f*rho*ux*ux*uy - 1.0f/12.0f*rho*ux*ux*uz - 1.0f/12.0f*rho*ux*uy*uy - 1.0f/4.0f*rho*ux*uy*uz - 1.0f/12.0f*rho*ux*uz*uz - 1.0f/12.0f*rho*uy*uy*uz - 1.0f/12.0f*rho*uy*uz*uz;
        case 26:
            return (1.0f/72.0f)*pxx + (1.0f/24.0f)*pxy + (1.0f/24.0f)*pxz + (1.0f/72.0f)*pyy + (1.0f/24.0f)*pyz + (1.0f/72.0f)*pzz + (1.0f/216.0f)*drho - 1.0f/72.0f*jx - 1.0f/72.0f*jy - 1.0f/72.0f*jz - 1.0f/24.0f*uy*pxx - 1.0f/24.0f*uz*pxx - 1.0f/12.0f*ux*pxy - 1.0f/12.0f*uy*pxy - 1.0f/8.0f*uz*pxy - 1.0f/12.0f*ux*pxz - 1.0f/8.0f*uy*pxz - 1.0f/12.0f*uz*pxz - 1.0f/24.0f*ux*pyy - 1.0f/24.0f*uz*pyy - 1.0f/8.0f*ux*pyz - 1.0f/12.0f*uy*pyz - 1.0f/12.0f*uz*pyz - 1.0f/24.0f*ux*pzz - 1.0f/24.0f*uy*pzz + (1.0f/12.0f)*rho*ux*ux*uy + (1.0f/12.0f)*rho*ux*ux*uz + (1.0f/12.0f)*rho*ux*uy*uy + (1.0f/4.0f)*rho*ux*uy*uz + (1.0f/12.0f)*rho*ux*uz*uz + (1.0f/12.0f)*rho*uy*uy*uz + (1.0f/12.0f)*rho*uy*uz*uz;
    }
    return 0;
}

}
