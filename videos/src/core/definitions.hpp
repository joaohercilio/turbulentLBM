#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


using dfloat = float;

enum Moment : int { DRHO, JX, JY, JZ, PXX, PXY, PXZ, PYY, PYZ, PZZ, NUMBER_OF_MOMENTS };

constexpr dfloat RHO_0 {1};
constexpr dfloat as2 {3};

constexpr int BLOCK_SIZE_X {64};
constexpr int BLOCK_SIZE_Y {4};
constexpr int BLOCK_SIZE_Z {1}; // <- THIS VALUE IS MANDATORY
constexpr int BLOCK_VOL {BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z};

constexpr int HX = BLOCK_SIZE_X + 2;
constexpr int HY = BLOCK_SIZE_Y + 2;
constexpr int HZ = BLOCK_SIZE_Z + 2;
constexpr int HALO_VOL = HX * HY * HZ;

#define cuda_CHECK(cmd)                                                \
    do {                                                               \
        const auto status = (cmd);                                     \
        if (status != cudaSuccess) {                                   \
            std::printf("cuda error %s:%d: %s\n", __FILE__, __LINE__,  \
            cudaGetErrorString(status));                               \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)
