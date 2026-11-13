#pragma once


#include "definitions.hpp"


constexpr int RED_THREADS = 256;
constexpr int RED_BLOCKS  = 1024;


static __global__ void reduceStatistics(const dfloat* m, double* partialK, double* partialEps, size_t numberOfNodes, dfloat TAU)
{
    double k = 0.0;
    double eps = 0.0;

    const double nu = (1.0/as2) * (TAU - 0.5);

    const size_t stride = static_cast<size_t>(gridDim.x) * RED_THREADS;
    for (size_t i = blockIdx.x * RED_THREADS + threadIdx.x; i < numberOfNodes; i += stride)
    {
        const size_t base = (i / BLOCK_VOL) * BLOCK_VOL * NUMBER_OF_MOMENTS + i % BLOCK_VOL;
        const double invRho = 1.0 / (RHO_0 + static_cast<double>(m[base + DRHO * BLOCK_VOL]));
        const double ux = m[base + JX * BLOCK_VOL] * invRho;
        const double uy = m[base + JY * BLOCK_VOL] * invRho;
        const double uz = m[base + JZ * BLOCK_VOL] * invRho;
        k += 0.5 * (ux*ux + uy*uy + uz*uz);

        const double mxx = m[base + PXX * BLOCK_VOL] * invRho;
        const double mxy = m[base + PXY * BLOCK_VOL] * invRho;
        const double mxz = m[base + PXZ * BLOCK_VOL] * invRho;
        const double myy = m[base + PYY * BLOCK_VOL] * invRho;
        const double myz = m[base + PYZ * BLOCK_VOL] * invRho;
        const double mzz = m[base + PZZ * BLOCK_VOL] * invRho;  

        //m = ux*uy   +   viscous * sxy;
        //sxy = 1/viscous * (m - ux*uy) = (as2/2.0) / (1.0f - TAU) * (m - rho*ux*uy)

        const double factor = as2 / (2.0 * (1.0 - TAU));

        const double sxx = factor * (mxx - ux * ux);
        const double syy = factor * (myy - uy * uy);
        const double szz = factor * (mzz - uz * uz);
        const double sxy = factor * (mxy - ux * uy);
        const double sxz = factor * (mxz - ux * uz);
        const double syz = factor * (myz - uy * uz);
        
        const double ss = sxx*sxx + syy*syy + szz*szz + 2.0*(sxy*sxy + sxz*sxz + syz*syz);

        eps += 2.0 * nu * ss ;
    }


    for (int o = 16; o > 0; o >>= 1)
    {
        k   += __shfl_down_sync(0xffffffff, k, o);
        eps += __shfl_down_sync(0xffffffff, eps, o);
    }


    __shared__ double warpK[RED_THREADS / 32];
    __shared__ double warpEps[RED_THREADS / 32];


    if ((threadIdx.x & 31) == 0)
    {
        const int warp = threadIdx.x >> 5;
        warpK[warp] = k;
        warpEps[warp] = eps;
    }

    __syncthreads();

    if (threadIdx.x < 32)
    {
        const bool valid = threadIdx.x < RED_THREADS / 32;

        k   = valid ? warpK[threadIdx.x]   : 0.0;
        eps = valid ? warpEps[threadIdx.x] : 0.0;

        for (int o = 16; o > 0; o >>= 1)
        {
            k   += __shfl_down_sync(0xffffffff, k, o);
            eps += __shfl_down_sync(0xffffffff, eps, o);
        }

        if (threadIdx.x == 0)
        {
            partialK[blockIdx.x] = k;
            partialEps[blockIdx.x] = eps;
        }
    }

}