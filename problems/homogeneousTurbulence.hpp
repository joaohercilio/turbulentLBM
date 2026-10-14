#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

#include "../core/definitions.hpp"
#include "../core/lbm.hpp"

static const char* NAMES[10] = {"drho","ux","uy","uz","sxx","sxy","sxz","syy","syz","szz"};

static void readVTK(std::vector<float> F[10], const size_t numberOfPoints)
{
    std::ifstream f("initial.vtk", std::ios::binary);
    std::string line;

    for (int c = 0; c < 10; c++) {
        const std::string header = std::string(NAMES[c]) + " ";
        while (std::getline(f, line) && line.rfind(header, 0) != 0) {}

        F[c].resize(numberOfPoints);
        f.read((char*)F[c].data(), numberOfPoints*sizeof(float));
        for (auto& v : F[c]) {
            uint32_t i; std::memcpy(&i, &v, 4);
            i = __builtin_bswap32(i);
            std::memcpy(&v, &i, 4);
        }
    }
}

struct InitFields { dfloat* p[10]; };

static __global__ void setInitialConditionsFromFile(dfloat* moments, InitFields f, int NX, int NY, int NZ, dfloat TAU)
{
    const int x = blockIdx.x*BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y*BLOCK_SIZE_Y + threadIdx.y;
    const int z = blockIdx.z*BLOCK_SIZE_Z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    const int i = x + NX*(y + NY*z);

    initMoments(moments, f.p[0][i], f.p[1][i], f.p[2][i], f.p[3][i], f.p[4][i], f.p[5][i], f.p[6][i], f.p[7][i], f.p[8][i], f.p[9][i],
                NX, NY, NZ, TAU);
}

inline void loadInitialConditions(dfloat* moments, dim3 grid, dim3 block,
                                  int NX, int NY, int NZ, dfloat TAU)
{
    const size_t n = static_cast<size_t>(NX) * NY * NZ;

    std::vector<float> F[10];
    readVTK(F, n);

    InitFields fields;
    for (int c = 0; c < 10; c++) {
        std::vector<dfloat> tmp(F[c].begin(), F[c].end());
        cuda_CHECK(cudaMalloc(&fields.p[c], n * sizeof(dfloat)));
        cuda_CHECK(cudaMemcpy(fields.p[c], tmp.data(), n * sizeof(dfloat), cudaMemcpyHostToDevice));
    }

    setInitialConditionsFromFile<<<grid, block>>>(moments, fields, NX, NY, NZ, TAU);
    cuda_CHECK(cudaGetLastError());
    cuda_CHECK(cudaDeviceSynchronize());

    for (int c = 0; c < 10; c++) cudaFree(fields.p[c]);
}
