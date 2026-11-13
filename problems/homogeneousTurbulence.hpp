#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <type_traits>
#include <hdf5.h>

#include "../core/definitions.hpp"
#include "../core/lbm.hpp"
#include "../core/output.hpp"
#include "../core/input.hpp"

static const char* NAMES[10] = {"drho","ux","uy","uz","sxx","sxy","sxz","syy","syz","szz"};

struct InitFields { dfloat* p[10]; };

static __global__ void setInitialConditionsFromFile(dfloat* moments, InitFields f, int NX, int NY, int NZ, dfloat TAU)
{
    const int x = blockIdx.x*BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y*BLOCK_SIZE_Y + threadIdx.y;
    const int z = blockIdx.z*BLOCK_SIZE_Z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    const size_t i = (static_cast<size_t>(z) * NY + y) * NX + x;

    initMoments(moments, f.p[0][i], f.p[1][i], f.p[2][i], f.p[3][i], f.p[4][i], f.p[5][i], f.p[6][i], f.p[7][i], f.p[8][i], f.p[9][i], NX, NY, NZ, TAU);
}

inline void loadInitialConditions(dfloat* moments, dim3 grid, dim3 block,
                                  int NX, int NY, int NZ, dfloat TAU, double scale)
{
    const size_t n = static_cast<size_t>(NX) * NY * NZ;
    const double delta = 2.0 * M_PI / NX;

    std::vector<dfloat> F(n);
    InitFields fields;
    hid_t file = H5Fopen("initial.vtkhdf", H5F_ACC_RDONLY, H5P_DEFAULT);
    for (int c = 0; c < 10; c++) {
        readPointData(file, NAMES[c], F.data(), NX, NY, NZ);
        const double factor = c == 0 ? scale * scale : c < 4 ? scale : scale * delta;
        for (auto& v : F) v *= factor;
        cuda_CHECK(cudaMalloc(&fields.p[c], n * sizeof(dfloat)));
        cuda_CHECK(cudaMemcpy(fields.p[c], F.data(), n * sizeof(dfloat), cudaMemcpyHostToDevice));
    }
    H5Fclose(file);

    setInitialConditionsFromFile<<<grid, block>>>(moments, fields, NX, NY, NZ, TAU);
    cuda_CHECK(cudaGetLastError());
    cuda_CHECK(cudaDeviceSynchronize());

    for (int c = 0; c < 10; c++) cudaFree(fields.p[c]);
}


constexpr int NX {512};
constexpr int NY {512};
constexpr int NZ {512};

constexpr double U_SCALE {1};

const dfloat TAU {static_cast<dfloat>(0.5 + U_SCALE * (readParameter("tau") - 0.5))};

const double TE {readParameter("te_lbm", "reference.txt") / U_SCALE};
const double DT_OUT {40 * TE / 199};
const int N_STEPS {static_cast<int>(std::ceil(5 * TE))};


inline void initialize(dfloat* moments, dim3 grid, dim3 block)
{
    loadInitialConditions(moments, grid, block, NX, NY, NZ, TAU, U_SCALE);
}


inline void boundary(dfloat*) {}


inline void output(const dfloat* moments, int iout, double t)
{
    writeFields(moments, iout, t, NX, NY, NZ, U_SCALE);
}
