#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <hdf5.h>

#include "definitions.hpp"
#include "index.hpp"


static __global__ void velocityKernel(const dfloat* m, float* u, const int NX, const int NY, const int NZ)
{
    const int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int z = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;
    const size_t n = static_cast<size_t>(NX) * NY * NZ;
    const size_t base = momIdxGlobal(0, x, y, z, NX, NY, NZ);
    const size_t i = (static_cast<size_t>(x) * NY + y) * NZ + z;
    const dfloat invRho = 1 / (RHO_0 + m[base + DRHO * BLOCK_VOL]);
    u[i]         = m[base + JX * BLOCK_VOL] * invRho;
    u[i + n]     = m[base + JY * BLOCK_VOL] * invRho;
    u[i + 2 * n] = m[base + JZ * BLOCK_VOL] * invRho;
}


inline void writeFields(const dfloat* moments, int iout, double t, const int NX, const int NY, const int NZ, double scale)
{
    const size_t n = static_cast<size_t>(NX) * NY * NZ;

    float* du = nullptr;
    cuda_CHECK(cudaMalloc(&du, 3 * n * sizeof(float)));
    velocityKernel<<<dim3(NX / BLOCK_SIZE_X, NY / BLOCK_SIZE_Y, NZ / BLOCK_SIZE_Z), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)>>>(moments, du, NX, NY, NZ);
    cuda_CHECK(cudaGetLastError());
    std::vector<float> u(3 * n);
    cuda_CHECK(cudaMemcpy(u.data(), du, 3 * n * sizeof(float), cudaMemcpyDeviceToHost));
    cuda_CHECK(cudaFree(du));

    hid_t file;
    if (iout == 0)
    {
        file = H5Fcreate("fields.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hid_t group = H5Gcreate(file, "u", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(H5Gcreate(group, "3D", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

        const long long rank = 1;
        const long long shape[3] = {NX, NY, NZ};
        const hsize_t three = 3;
        hid_t scalar = H5Screate(H5S_SCALAR);
        hid_t vec = H5Screate_simple(1, &three, nullptr);
        hid_t root = H5Acreate(file, "scale", H5T_NATIVE_DOUBLE, scalar, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(root, H5T_NATIVE_DOUBLE, &scale);
        H5Aclose(root);
        hid_t attr = H5Acreate(group, "rank", H5T_NATIVE_LLONG, scalar, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_LLONG, &rank);
        H5Aclose(attr);
        attr = H5Acreate(group, "shape", H5T_NATIVE_LLONG, vec, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_LLONG, shape);
        H5Aclose(attr);
        H5Sclose(scalar);
        H5Sclose(vec);

        hid_t mesh = H5Gcreate(group, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        const int N[3] = {NX, NY, NZ};
        for (int d = 0; d < 3; ++d)
        {
            std::vector<double> x(N[d]);
            for (int j = 0; j < N[d]; ++j) x[j] = 2.0 * M_PI * j / N[d];
            const hsize_t len = N[d];
            hid_t space = H5Screate_simple(1, &len, nullptr);
            hid_t dset = H5Dcreate(mesh, ("x" + std::to_string(d)).c_str(), H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x.data());
            H5Dclose(dset);
            H5Sclose(space);
        }
        H5Gclose(mesh);
        H5Gclose(group);
    }
    else
    {
        file = H5Fopen("fields.h5", H5F_ACC_RDWR, H5P_DEFAULT);
    }

    const hsize_t dims[4] = {3, static_cast<hsize_t>(NX), static_cast<hsize_t>(NY), static_cast<hsize_t>(NZ)};
    hid_t space = H5Screate_simple(4, dims, nullptr);
    hid_t dset = H5Dcreate(file, ("u/3D/" + std::to_string(iout)).c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, u.data());
    H5Dclose(dset);
    H5Sclose(space);
    H5Fclose(file);

    std::ofstream g("times.dat", iout == 0 ? std::ios::trunc : std::ios::app);
    g << iout << '\t' << std::fixed << std::setprecision(10) << t << '\n';
}
