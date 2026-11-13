#pragma once


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include <hdf5.h>


#include "../core/definitions.hpp"
#include "../core/index.hpp"
#include "../core/lbm.hpp"
#include "../core/input.hpp"

constexpr int NX {256};
constexpr int NY {768};
constexpr int NZ {154};

constexpr double U_SCALE {1};


const dfloat TAU {static_cast<dfloat>(readParameter("tau"))};
const double T_REF {readParameter("delta_w") / readParameter("U")};
const double DT_OUT {T_REF};
const int N_STEPS {static_cast<int>(std::ceil(20 * T_REF))};


struct Fields { dfloat* p[10]; };

static __global__ void setInitialConditions(dfloat* moments, Fields f, dfloat TAU)
{
    const int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int z = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;

    const size_t i = (static_cast<size_t>(z) * NY + y) * NX + x;

    initMoments(moments, f.p[0][i], f.p[1][i], f.p[2][i], f.p[3][i], f.p[4][i], f.p[5][i], f.p[6][i], f.p[7][i], f.p[8][i], f.p[9][i],
                NX, NY, NZ, TAU);
}


static __global__ void freeSlip(dfloat* moments)
{
    const int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int z = blockIdx.z;

    const size_t ghost = momIdxGlobal(0, x, blockIdx.y == 0 ? 0 : NY - 1, z, NX, NY, NZ);
    const size_t fluid = momIdxGlobal(0, x, blockIdx.y == 0 ? 1 : NY - 2, z, NX, NY, NZ);

    for (int i = 0; i < NUMBER_OF_MOMENTS; ++i)
        moments[ghost + i * BLOCK_VOL] = (i == JY || i == PXY || i == PYZ ? -1 : 1) * moments[fluid + i * BLOCK_VOL];
}


inline void boundary(dfloat* moments)
{
    freeSlip<<<dim3(NX / BLOCK_SIZE_X, 2, NZ), BLOCK_SIZE_X>>>(moments);
}


inline void initialize(dfloat* moments, dim3 grid, dim3 block)
{
    if (readParameter("NX") != NX || readParameter("NY") != NY || readParameter("NZ") != NZ)
    {
        std::printf("initial.txt nao corresponde a NX, NY, NZ\n");
        std::exit(EXIT_FAILURE);
    }

    const char* names[10] = {"drho", "ux", "uy", "uz", "sxx", "sxy", "sxz", "syy", "syz", "szz"};
    const size_t n = static_cast<size_t>(NX) * NY * NZ;

    std::vector<dfloat> h(n);
    Fields f;
    hid_t file = H5Fopen("initial.vtkhdf", H5F_ACC_RDONLY, H5P_DEFAULT);
    for (int c = 0; c < 10; ++c)
    {
        readPointData(file, names[c], h.data(), NX, NY, NZ);
        cuda_CHECK(cudaMalloc(&f.p[c], n * sizeof(dfloat)));
        cuda_CHECK(cudaMemcpy(f.p[c], h.data(), n * sizeof(dfloat), cudaMemcpyHostToDevice));
    }
    H5Fclose(file);

    setInitialConditions<<<grid, block>>>(moments, f, TAU);
    cuda_CHECK(cudaDeviceSynchronize());
    for (int c = 0; c < 10; ++c) cudaFree(f.p[c]);

    boundary(moments);
}



static __global__ void velocity(const dfloat* m, float* u)
{
    const int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int z = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;

    const size_t base = momIdxGlobal(0, x, y, z, NX, NY, NZ);
    const size_t i = 3 * ((static_cast<size_t>(z) * NY + y) * NX + x);
    const dfloat inv = 1 / (RHO_0 + m[base + DRHO * BLOCK_VOL]);

    u[i]     = m[base + JX * BLOCK_VOL] * inv;
    u[i + 1] = m[base + JY * BLOCK_VOL] * inv;
    u[i + 2] = m[base + JZ * BLOCK_VOL] * inv;
}


inline void attributes(hid_t group, const char* name, hid_t type, hsize_t n, const void* data)
{
    hid_t space = H5Screate_simple(1, &n, nullptr);
    hid_t attr = H5Acreate(group, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type, data);
    H5Aclose(attr);
    H5Sclose(space);
}


inline void output(const dfloat* moments, int iout, double)
{
    const size_t n = static_cast<size_t>(NX) * NY * NZ;
    float* d;
    cuda_CHECK(cudaMalloc(&d, 3 * n * sizeof(float)));
    velocity<<<dim3(NX / BLOCK_SIZE_X, NY / BLOCK_SIZE_Y, NZ / BLOCK_SIZE_Z), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)>>>(moments, d);
    std::vector<float> u(3 * n);
    cuda_CHECK(cudaMemcpy(u.data(), d, 3 * n * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_CHECK(cudaFree(d));

    char name[32];
    std::snprintf(name, sizeof(name), "fields_%04d.vtkhdf", iout);
    hid_t file = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t root = H5Gcreate(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    const int version[2] = {1, 0};
    const long long extent[6] = {0, NX - 1, 0, NY - 1, 0, NZ - 1};
    const double origin[3] = {0, 0, 0}, spacing[3] = {1, 1, 1}, direction[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    attributes(root, "Version", H5T_NATIVE_INT, 2, version);
    attributes(root, "WholeExtent", H5T_NATIVE_LLONG, 6, extent);
    attributes(root, "Origin", H5T_NATIVE_DOUBLE, 3, origin);
    attributes(root, "Spacing", H5T_NATIVE_DOUBLE, 3, spacing);
    attributes(root, "Direction", H5T_NATIVE_DOUBLE, 9, direction);
    hid_t str = H5Tcopy(H5T_C_S1);
    H5Tset_size(str, 9);
    hid_t scalar = H5Screate(H5S_SCALAR);
    hid_t type = H5Acreate(root, "Type", str, scalar, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(type, str, "ImageData");
    H5Aclose(type);
    H5Sclose(scalar);
    H5Tclose(str);

    hid_t point = H5Gcreate(root, "PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const hsize_t dims[4] = {NZ, NY, NX, 3};
    hid_t space = H5Screate_simple(4, dims, nullptr);
    hid_t dset = H5Dcreate(point, "velocity", H5T_IEEE_F32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, u.data());
    H5Dclose(dset);
    H5Sclose(space);
    H5Gclose(point);
    H5Gclose(root);
    H5Fclose(file);
}



/*
    TEMPORAL MIXING LAYER, CASE ROLLUP OF ROGERS & MOSER (1992), J. FLUID MECH. 243, 183-226
    initial condition generated by ML/initial.py (initial.vtkhdf + initial.txt, lattice units)
    u = U erf(sqrt(pi) y / delta_w), Re = U delta_w / nu = 500, Lx = 1.16 (2 pi) delta_w, Lz = 0.6 Lx
    rows y = 0 and y = NY - 1 are ghosts mirrored from the fluid rows: free-slip walls at y = 0.5 and y = NY - 1.5
    time unit T_REF = delta_w / U, output every T_REF up to 20 T_REF
    output: fields_XXXX.vtkhdf (velocity, lattice units), opened directly as a time series in ParaView
*/

