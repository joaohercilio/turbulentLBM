#pragma once

#include "../core/definitions.hpp"

namespace d3q27 {


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