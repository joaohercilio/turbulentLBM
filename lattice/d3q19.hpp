#pragma once

#include "../core/definitions.hpp"

namespace d3q19 {


constexpr int NUMBER_OF_VELOCITIES {19};

__device__ __constant__ constexpr int cx[19] = { 0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0 };
__device__ __constant__ constexpr int cy[19] = { 0,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0, -1,  1, -1,  1 };
__device__ __constant__ constexpr int cz[19] = { 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1, -1,  1, -1,  1,  1, -1 };


static __device__ __forceinline__ dfloat reconstructPopulation(
    int k, dfloat drho, dfloat jx, dfloat jy, dfloat jz,
    dfloat pxx, dfloat pxy, dfloat pxz,
    dfloat pyy, dfloat pyz, dfloat pzz)
{
    switch (k) {
        
        case 0:
            return (1.0f/3.0f)*drho - 1.0f/2.0f*pxx - 1.0f/2.0f*pyy - 1.0f/2.0f*pzz;
        case 1:
            return (1.0f/18.0f)*drho + (1.0f/6.0f)*jx + (1.0f/6.0f)*pxx - 1.0f/12.0f*pyy - 1.0f/12.0f*pzz;
        case 2:
            return (1.0f/18.0f)*drho - 1.0f/6.0f*jx + (1.0f/6.0f)*pxx - 1.0f/12.0f*pyy - 1.0f/12.0f*pzz;
        case 3:
            return (1.0f/18.0f)*drho + (1.0f/6.0f)*jy - 1.0f/12.0f*pxx + (1.0f/6.0f)*pyy - 1.0f/12.0f*pzz;
        case 4:
            return (1.0f/18.0f)*drho - 1.0f/6.0f*jy - 1.0f/12.0f*pxx + (1.0f/6.0f)*pyy - 1.0f/12.0f*pzz;
        case 5:
            return (1.0f/18.0f)*drho + (1.0f/6.0f)*jz - 1.0f/12.0f*pxx - 1.0f/12.0f*pyy + (1.0f/6.0f)*pzz;
        case 6:
            return (1.0f/18.0f)*drho - 1.0f/6.0f*jz - 1.0f/12.0f*pxx - 1.0f/12.0f*pyy + (1.0f/6.0f)*pzz;
        case 7:
            return (1.0f/36.0f)*drho + (1.0f/12.0f)*jx + (1.0f/12.0f)*jy + (1.0f/12.0f)*pxx + (1.0f/4.0f)*pxy + (1.0f/12.0f)*pyy - 1.0f/24.0f*pzz;
        case 8:
            return (1.0f/36.0f)*drho - 1.0f/12.0f*jx - 1.0f/12.0f*jy + (1.0f/12.0f)*pxx + (1.0f/4.0f)*pxy + (1.0f/12.0f)*pyy - 1.0f/24.0f*pzz;
        case 9:
            return (1.0f/36.0f)*drho + (1.0f/12.0f)*jx - 1.0f/12.0f*jy + (1.0f/12.0f)*pxx - 1.0f/4.0f*pxy + (1.0f/12.0f)*pyy - 1.0f/24.0f*pzz;
        case 10:
            return (1.0f/36.0f)*drho - 1.0f/12.0f*jx + (1.0f/12.0f)*jy + (1.0f/12.0f)*pxx - 1.0f/4.0f*pxy + (1.0f/12.0f)*pyy - 1.0f/24.0f*pzz;
        case 11:
            return (1.0f/36.0f)*drho + (1.0f/12.0f)*jx + (1.0f/12.0f)*jz + (1.0f/12.0f)*pxx + (1.0f/4.0f)*pxz - 1.0f/24.0f*pyy + (1.0f/12.0f)*pzz;
        case 12:
            return (1.0f/36.0f)*drho - 1.0f/12.0f*jx - 1.0f/12.0f*jz + (1.0f/12.0f)*pxx + (1.0f/4.0f)*pxz - 1.0f/24.0f*pyy + (1.0f/12.0f)*pzz;
        case 13:
            return (1.0f/36.0f)*drho + (1.0f/12.0f)*jx - 1.0f/12.0f*jz + (1.0f/12.0f)*pxx - 1.0f/4.0f*pxz - 1.0f/24.0f*pyy + (1.0f/12.0f)*pzz;
        case 14:
            return (1.0f/36.0f)*drho - 1.0f/12.0f*jx + (1.0f/12.0f)*jz + (1.0f/12.0f)*pxx - 1.0f/4.0f*pxz - 1.0f/24.0f*pyy + (1.0f/12.0f)*pzz;
        case 15:
            return (1.0f/36.0f)*drho - 1.0f/12.0f*jy - 1.0f/12.0f*jz - 1.0f/24.0f*pxx + (1.0f/12.0f)*pyy + (1.0f/4.0f)*pyz + (1.0f/12.0f)*pzz;
        case 16:
            return (1.0f/36.0f)*drho + (1.0f/12.0f)*jy + (1.0f/12.0f)*jz - 1.0f/24.0f*pxx + (1.0f/12.0f)*pyy + (1.0f/4.0f)*pyz + (1.0f/12.0f)*pzz;
        case 17:
            return (1.0f/36.0f)*drho - 1.0f/12.0f*jy + (1.0f/12.0f)*jz - 1.0f/24.0f*pxx + (1.0f/12.0f)*pyy - 1.0f/4.0f*pyz + (1.0f/12.0f)*pzz;
        case 18:
            return (1.0f/36.0f)*drho + (1.0f/12.0f)*jy - 1.0f/12.0f*jz - 1.0f/24.0f*pxx + (1.0f/12.0f)*pyy - 1.0f/4.0f*pyz + (1.0f/12.0f)*pzz;
    }
    return 0;
}

}