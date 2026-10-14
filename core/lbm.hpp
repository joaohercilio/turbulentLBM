#pragma once


#include "definitions.hpp"
#include "index.hpp"

#include "../lattice/lattice.hpp"


static __device__ void initMoments(dfloat* m, dfloat drho, dfloat ux, dfloat uy, dfloat uz,
                            dfloat sxx, dfloat sxy, dfloat sxz, dfloat syy, dfloat syz, dfloat szz,
                            const int NX, const int NY, const int NZ, const dfloat TAU)
{
    const dfloat rho = RHO_0 + drho;
    m[momIdxGlobal(DRHO, NX, NY, NZ)] = drho;
    m[momIdxGlobal(JX, NX, NY, NZ)]  = rho * ux;
    m[momIdxGlobal(JY, NX, NY, NZ)]  = rho * uy;
    m[momIdxGlobal(JZ, NX, NY, NZ)]  = rho * uz;

    const dfloat viscous = (RHO_0 + drho) * 2.0f / as2 * (1.0f - TAU);

    m[momIdxGlobal(PXX, NX, NY, NZ)] = rho * ux * ux + viscous * sxx;
    m[momIdxGlobal(PXY, NX, NY, NZ)] = rho * ux * uy + viscous * sxy;
    m[momIdxGlobal(PXZ, NX, NY, NZ)] = rho * ux * uz + viscous * sxz;
    m[momIdxGlobal(PYY, NX, NY, NZ)] = rho * uy * uy + viscous * syy;
    m[momIdxGlobal(PYZ, NX, NY, NZ)] = rho * uy * uz + viscous * syz;
    m[momIdxGlobal(PZZ, NX, NY, NZ)] = rho * uz * uz + viscous * szz;
}



static __global__ void kernel
    (const dfloat* momentsA, dfloat* momentsB, 
    const int NX, const int NY, const int NZ,
    const dfloat OMEGA_EQ, const dfloat OMEGA_NEQ)
{
    __shared__ dfloat s[NUMBER_OF_MOMENTS][HZ][HY][HX];

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    const int tid = tx + BLOCK_SIZE_X * (ty + BLOCK_SIZE_Y * tz);

    for (int i = tid; i < HALO_VOL; i += BLOCK_VOL)
    {
        const int hx = i % HX;
        const int hy = (i / HX) % HY;
        const int hz = i / (HX * HY);

        const int gx = (bx * BLOCK_SIZE_X + hx - 1 + NX) % NX;
        const int gy = (by * BLOCK_SIZE_Y + hy - 1 + NY) % NY;
        const int gz = (bz * BLOCK_SIZE_Z + hz - 1 + NZ) % NZ;
        const size_t node = nodeIdxGlobal(gx, gy, gz, NX, NY, NZ);

        s[DRHO][hz][hy][hx] = momentsA[node + DRHO * BLOCK_VOL];
        s[JX  ][hz][hy][hx] = momentsA[node + JX  * BLOCK_VOL];
        s[JY  ][hz][hy][hx] = momentsA[node + JY  * BLOCK_VOL];
        s[JZ  ][hz][hy][hx] = momentsA[node + JZ  * BLOCK_VOL];
        s[PXX ][hz][hy][hx] = momentsA[node + PXX * BLOCK_VOL];
        s[PXY ][hz][hy][hx] = momentsA[node + PXY * BLOCK_VOL];
        s[PXZ ][hz][hy][hx] = momentsA[node + PXZ * BLOCK_VOL];
        s[PYY ][hz][hy][hx] = momentsA[node + PYY * BLOCK_VOL];
        s[PYZ ][hz][hy][hx] = momentsA[node + PYZ * BLOCK_VOL];
        s[PZZ ][hz][hy][hx] = momentsA[node + PZZ * BLOCK_VOL];
    }

    __syncthreads();

    dfloat drho {0}, jx {0}, jy {0}, jz {0}, pxx {0}, pxy {0}, pxz {0}, pyy {0}, pyz {0}, pzz {0};

    #pragma unroll
    for (int k = 0; k < NUMBER_OF_VELOCITIES; ++k)
    {
        const int sx = tx + 1 - cx[k];
        const int sy = ty + 1 - cy[k];
        const int sz = tz + 1 - cz[k];

        const dfloat fk = reconstructPopulation(k, s[DRHO][sz][sy][sx],
                                                   s[JX  ][sz][sy][sx], s[JY ][sz][sy][sx], s[JZ ][sz][sy][sx],
                                                   s[PXX ][sz][sy][sx], s[PXY][sz][sy][sx], s[PXZ][sz][sy][sx],
                                                   s[PYY ][sz][sy][sx], s[PYZ][sz][sy][sx], s[PZZ][sz][sy][sx]);

        drho += fk;
        jx   += fk * cx[k];
        jy   += fk * cy[k];
        jz   += fk * cz[k];
        pxx  += fk * (cx[k] * cx[k] - static_cast<dfloat>(1.0/3.0));
        pxy  += fk * cx[k] * cy[k];
        pxz  += fk * cx[k] * cz[k];
        pyy  += fk * (cy[k] * cy[k] - static_cast<dfloat>(1.0/3.0));
        pyz  += fk * cy[k] * cz[k];
        pzz  += fk * (cz[k] * cz[k] - static_cast<dfloat>(1.0/3.0));
    }


    // dfloat rhoI {0};

    // if (bz == 0)
    // {
    //     for (int k = 0; k < NUMBER_OF_VELOCITIES; ++k)
    //     {
    //         if ( cy[k] < 1 ) {

    //             const int sx = tx + 1 - cx[k];
    //             const int sy = ty + 1 - cy[k];
    //             const int sz = tz + 1 - cz[k];

    //             const dfloat fk = reconstructPopulation(k, s[DRHO][sz][sy][sx],
    //                                                s[JX  ][sz][sy][sx], s[JY ][sz][sy][sx], s[JZ ][sz][sy][sx],
    //                                                s[PXX ][sz][sy][sx], s[PXY][sz][sy][sx], s[PXZ][sz][sy][sx],
    //                                                s[PYY ][sz][sy][sx], s[PYZ][sz][sy][sx], s[PZZ][sz][sy][sx]);

    //             rhoI += fk + w[i];

    //         }
    //     }
    // }


    const dfloat rho = drho + RHO_0;
    const dfloat invRho = 1.0f / rho;


    pxx = OMEGA_NEQ * pxx + OMEGA_EQ * jx * jx * invRho;
    pxy = OMEGA_NEQ * pxy + OMEGA_EQ * jx * jy * invRho;
    pxz = OMEGA_NEQ * pxz + OMEGA_EQ * jx * jz * invRho;
    pyy = OMEGA_NEQ * pyy + OMEGA_EQ * jy * jy * invRho;
    pyz = OMEGA_NEQ * pyz + OMEGA_EQ * jy * jz * invRho;
    pzz = OMEGA_NEQ * pzz + OMEGA_EQ * jz * jz * invRho;

    const int x = bx * BLOCK_SIZE_X + tx;
    const int y = by * BLOCK_SIZE_Y + ty;
    const int z = bz * BLOCK_SIZE_Z + tz;
    
    const size_t base = momIdxGlobal(0, x, y, z, NX, NY, NZ);
    
    momentsB[base + DRHO * BLOCK_VOL] = drho;
    momentsB[base + JX  * BLOCK_VOL] = jx;
    momentsB[base + JY  * BLOCK_VOL] = jy;
    momentsB[base + JZ  * BLOCK_VOL] = jz;
    momentsB[base + PXX * BLOCK_VOL] = pxx;
    momentsB[base + PXY * BLOCK_VOL] = pxy;
    momentsB[base + PXZ * BLOCK_VOL] = pxz;
    momentsB[base + PYY * BLOCK_VOL] = pyy;
    momentsB[base + PYZ * BLOCK_VOL] = pyz;
    momentsB[base + PZZ * BLOCK_VOL] = pzz;
}
