#pragma once

// Variante de problems/shearLayer.hpp para o video dos ribs (condicao inicial: videos/initial.py).
// Diferencas: dominio 512 x 512 x 462 (2 lambda_x, 3 lambda_z), 26 T_REF, saida a cada T_REF/4
// so com Q-criterion e omega_x numa faixa central em y (o resto do dominio nao tem vorticidade).

#include <cmath>
#include <numbers>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include <hdf5.h>

#include "core/definitions.hpp"
#include "core/index.hpp"
#include "core/lbm.hpp"

constexpr int NX {512};
constexpr int NY {512};
constexpr int NZ {462};

constexpr int NYC {320};               // linhas gravadas em y, centradas na camada
constexpr int Y0 {(NY - NYC) / 2};

constexpr double U_SCALE {1};


// os mesmos parametros de videos/initial.py
constexpr int LX {NX / 2};             // lambda_x
constexpr double U {0.05};
constexpr double RE {1000};
constexpr double DELTA_W {LX / (1.16 * 2 * std::numbers::pi)};
constexpr double NU {U * DELTA_W / RE};

constexpr dfloat TAU {static_cast<dfloat>(0.5 + 3 * NU)};
constexpr double T_REF {DELTA_W / U};
const double DT_OUT {T_REF / 4};
const int N_STEPS {static_cast<int>(std::ceil(26 * T_REF))};


struct Fields { dfloat* p[10]; };

static __global__ void setInitialConditions(dfloat* moments, Fields f, dfloat TAU)
{
    const int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int z = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;

    const size_t i = (static_cast<size_t>(x) * NY + y) * NZ + z;

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
    const char* names[10] = {"drho", "ux", "uy", "uz", "sxx", "sxy", "sxz", "syy", "syz", "szz"};
    const size_t n = static_cast<size_t>(NX) * NY * NZ;
    const hid_t type = std::is_same_v<dfloat, float> ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;

    std::vector<dfloat> h(n);
    Fields f;
    hid_t file = H5Fopen("initial.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    for (int c = 0; c < 10; ++c)
    {
        hid_t dset = H5Dopen(file, names[c], H5P_DEFAULT);
        hid_t space = H5Dget_space(dset);
        hsize_t dims[3] = {0, 0, 0};
        H5Sget_simple_extent_dims(space, dims, nullptr);
        H5Sclose(space);
        if (dims[0] != hsize_t(NX) || dims[1] != hsize_t(NY) || dims[2] != hsize_t(NZ))
        {
            std::printf("initial.h5/%s nao tem shape [NX, NY, NZ] = [%d, %d, %d]\n", names[c], NX, NY, NZ);
            std::exit(EXIT_FAILURE);
        }
        H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, h.data());
        H5Dclose(dset);
        cuda_CHECK(cudaMalloc(&f.p[c], n * sizeof(dfloat)));
        cuda_CHECK(cudaMemcpy(f.p[c], h.data(), n * sizeof(dfloat), cudaMemcpyHostToDevice));
    }
    H5Fclose(file);

    setInitialConditions<<<grid, block>>>(moments, f, TAU);
    cuda_CHECK(cudaDeviceSynchronize());
    for (int c = 0; c < 10; ++c) cudaFree(f.p[c]);

    boundary(moments);
}


__device__ __forceinline__ void vel(const dfloat* m, int x, int y, int z, float u[3])
{
    x = (x + NX) % NX;
    z = (z + NZ) % NZ;
    const size_t b = momIdxGlobal(0, x, y, z, NX, NY, NZ);
    const dfloat inv = 1 / (RHO_0 + m[b + DRHO * BLOCK_VOL]);
    u[0] = m[b + JX * BLOCK_VOL] * inv;
    u[1] = m[b + JY * BLOCK_VOL] * inv;
    u[2] = m[b + JZ * BLOCK_VOL] * inv;
}


// Q = (|Omega|^2 - |S|^2)/2 e omega_x, por diferencas centradas, na faixa y = Y0 .. Y0 + NYC - 1.
// Normalizados por (U/delta_w)^2 e U/delta_w.
static __global__ void criteria(const dfloat* m, float* q, float* wx, float qs, float ws)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int yc = blockIdx.y;
    const int z = blockIdx.z;
    const int y = Y0 + yc;

    float a[3], b[3], g[3][3];
    vel(m, x + 1, y, z, a); vel(m, x - 1, y, z, b);
    for (int i = 0; i < 3; ++i) g[i][0] = 0.5f * (a[i] - b[i]);
    vel(m, x, y + 1, z, a); vel(m, x, y - 1, z, b);
    for (int i = 0; i < 3; ++i) g[i][1] = 0.5f * (a[i] - b[i]);
    vel(m, x, y, z + 1, a); vel(m, x, y, z - 1, b);
    for (int i = 0; i < 3; ++i) g[i][2] = 0.5f * (a[i] - b[i]);

    float s2 = 0, o2 = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const float s = 0.5f * (g[i][j] + g[j][i]), o = 0.5f * (g[i][j] - g[j][i]);
            s2 += s * s;
            o2 += o * o;
        }

    const size_t i = (static_cast<size_t>(z) * NYC + yc) * NX + x;
    q[i] = 0.5f * (o2 - s2) * qs;
    wx[i] = (g[2][1] - g[1][2]) * ws;
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
    const size_t n = static_cast<size_t>(NX) * NYC * NZ;
    static float* d = nullptr;
    if (!d) cuda_CHECK(cudaMalloc(&d, 2 * n * sizeof(float)));

    const double s = T_REF;
    criteria<<<dim3(NX / 64, NYC, NZ), 64>>>(moments, d, d + n, static_cast<float>(s * s), static_cast<float>(s));
    std::vector<float> h(2 * n);
    cuda_CHECK(cudaMemcpy(h.data(), d, 2 * n * sizeof(float), cudaMemcpyDeviceToHost));

    char name[32];
    std::snprintf(name, sizeof(name), "fields_%04d.vtkhdf", iout);
    hid_t file = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t root = H5Gcreate(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    const int version[2] = {1, 0};
    const long long extent[6] = {0, NX - 1, 0, NYC - 1, 0, NZ - 1};
    const double origin[3] = {0, Y0, 0}, spacing[3] = {1, 1, 1}, direction[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
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
    const hsize_t dims[3] = {NZ, NYC, NX};
    hid_t space = H5Screate_simple(3, dims, nullptr);
    const char* names[2] = {"Q", "omega_x"};
    for (int c = 0; c < 2; ++c)
    {
        hid_t dset = H5Dcreate(point, names[c], H5T_IEEE_F32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, h.data() + c * n);
        H5Dclose(dset);
    }
    H5Sclose(space);
    H5Gclose(point);
    H5Gclose(root);
    H5Fclose(file);
}
