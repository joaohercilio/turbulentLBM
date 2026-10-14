#pragma once


#include "definitions.hpp"


__device__ __forceinline__ size_t nodeIdxGlobal(int x, int y, int z, const int NX, const int NY, const int NZ)
{
    const int bx = x / BLOCK_SIZE_X, tx = x % BLOCK_SIZE_X;
    const int by = y / BLOCK_SIZE_Y, ty = y % BLOCK_SIZE_Y;
    const int bz = z / BLOCK_SIZE_Z, tz = z % BLOCK_SIZE_Z;

    size_t block = bx + (NX / BLOCK_SIZE_X) * (by + (NY / BLOCK_SIZE_Y) * bz);
    const int threadInBlock = tx + BLOCK_SIZE_X * (ty + BLOCK_SIZE_Y * tz);
    return block * BLOCK_VOL * NUMBER_OF_MOMENTS + threadInBlock;
}



__device__ __forceinline__ size_t momIdxGlobal(int momIdx, int x, int y, int z, const int NX, const int NY, const int NZ)
{
    const int bx = x / BLOCK_SIZE_X, tx = x % BLOCK_SIZE_X;
    const int by = y / BLOCK_SIZE_Y, ty = y % BLOCK_SIZE_Y;
    const int bz = z / BLOCK_SIZE_Z, tz = z % BLOCK_SIZE_Z;

    size_t block = bx + (NX / BLOCK_SIZE_X) * (by + (NY / BLOCK_SIZE_Y) * bz);
    int threadInBlock = tx + BLOCK_SIZE_X * (ty + BLOCK_SIZE_Y * tz);
    return block * BLOCK_VOL * NUMBER_OF_MOMENTS + momIdx * BLOCK_VOL + threadInBlock;
}



__device__ __forceinline__ size_t momIdxGlobal(int momIdx, const int NX, const int NY, const int NZ)
{
    const int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int z = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;
    return momIdxGlobal(momIdx, x, y, z, NX, NY, NZ);
}
