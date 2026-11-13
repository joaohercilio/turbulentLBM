#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>

#include "core/definitions.hpp"
#include "core/lbm.hpp"
#include "problems/homogeneousTurbulence.hpp"

constexpr int NS {512};
constexpr int OUT_EVERY {85};

static __device__ void velocity(const dfloat* m, int x, int y, int z, float* u)
{
    x = (x + NX) % NX; y = (y + NY) % NY; z = (z + NZ) % NZ;
    const size_t base = momIdxGlobal(0, x, y, z, NX, NY, NZ);
    const dfloat inv = 1 / (RHO_0 + m[base + DRHO * BLOCK_VOL]);
    u[0] = m[base + JX * BLOCK_VOL] * inv;
    u[1] = m[base + JY * BLOCK_VOL] * inv;
    u[2] = m[base + JZ * BLOCK_VOL] * inv;
}

static __device__ void gradient(const dfloat* m, int x, int y, int z, float* omega, float* wmag)
{
    float g[3][3], up[3], um[3];
    const int d[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int j = 0; j < 3; ++j)
    {
        velocity(m, x + d[j][0], y + d[j][1], z + d[j][2], up);
        velocity(m, x - d[j][0], y - d[j][1], z - d[j][2], um);
        for (int i = 0; i < 3; ++i) g[i][j] = 0.5f * (up[i] - um[i]);
    }
    float a2 = 0, b2 = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const float s = 0.5f * (g[i][j] + g[j][i]), r = 0.5f * (g[i][j] - g[j][i]);
            a2 += s * s; b2 += r * r;
        }
    *omega = b2 / (a2 + b2 + 1e-14f);
    const float wx = g[2][1] - g[1][2], wy = g[0][2] - g[2][0], wz = g[1][0] - g[0][1];
    *wmag = sqrtf(wx * wx + wy * wy + wz * wz);
}

static __global__ void cube(const dfloat* m, float* om, float* wm)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<size_t>(NS) * NS * NS) return;
    gradient(m, i % NS, (i / NS) % NS, i / (NS * NS), &om[i], &wm[i]);
}

static __global__ void quantize(const float* om, const float* wm, unsigned char* q, float wmax)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t nc = static_cast<size_t>(NS) * NS * NS;
    if (i >= nc) return;
    q[i] = static_cast<unsigned char>(fminf(255.0f, rintf(255.0f * om[i])));
    q[i + nc] = static_cast<unsigned char>(fminf(255.0f, rintf(255.0f * wm[i] / wmax)));
}

int main()
{
    cuda_CHECK(cudaSetDevice(0));
    const dfloat OMEGA_EQ = 1 / TAU, OMEGA_NEQ = 1 - OMEGA_EQ;
    const size_t n = static_cast<size_t>(NX) * NY * NZ * NUMBER_OF_MOMENTS;
    const size_t nc = static_cast<size_t>(NS) * NS * NS;
    dim3 block {BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z}, grid {NX / BLOCK_SIZE_X, NY / BLOCK_SIZE_Y, NZ / BLOCK_SIZE_Z};
    dfloat *A, *B;
    float *om, *wm;
    unsigned char* q;
    cuda_CHECK(cudaMalloc(&A, n * sizeof(dfloat)));
    cuda_CHECK(cudaMalloc(&B, n * sizeof(dfloat)));
    cuda_CHECK(cudaMalloc(&om, nc * sizeof(float)));
    cuda_CHECK(cudaMalloc(&wm, nc * sizeof(float)));
    cuda_CHECK(cudaMalloc(&q, 2 * nc));
    initialize(A, grid, block);
    std::vector<float> h(nc);
    std::vector<unsigned char> hq(2 * nc);
    cube<<<(nc + 255) / 256, 256>>>(A, om, wm);
    cuda_CHECK(cudaMemcpy(h.data(), wm, nc * sizeof(float), cudaMemcpyDeviceToHost));
    double w2 = 0;
    for (float v : h) w2 += static_cast<double>(v) * v;
    const float wmax = 5 * std::sqrt(w2 / nc);
    std::cout << "omega_rms0 " << wmax / 5 << std::endl;
    std::ofstream times("times.dat");
    std::cout << "tau " << TAU << " steps " << N_STEPS << " te " << TE << std::endl;
    for (int t = 0, iout = 0; t <= N_STEPS; ++t)
    {
        if (t % OUT_EVERY == 0)
        {
            cube<<<(nc + 255) / 256, 256>>>(A, om, wm);
            quantize<<<(nc + 255) / 256, 256>>>(om, wm, q, wmax);
            cuda_CHECK(cudaMemcpy(hq.data(), q, 2 * nc, cudaMemcpyDeviceToHost));
            const std::string name = "frames/" + std::to_string(iout);
            std::ofstream(name + ".tmp", std::ios::binary).write(reinterpret_cast<char*>(hq.data()), 2 * nc);
            std::rename((name + ".tmp").c_str(), (name + ".bin").c_str());
            times << iout++ << '\t' << t / TE << std::endl;
        }
        kernel<<<grid, block>>>(A, B, NX, NY, NZ, OMEGA_EQ, OMEGA_NEQ);
        std::swap(A, B);
    }
    cuda_CHECK(cudaDeviceSynchronize());
    cuda_CHECK(cudaGetLastError());
    return 0;
}
