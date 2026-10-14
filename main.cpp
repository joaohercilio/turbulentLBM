#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>

#include "core/definitions.hpp"
#include "core/statistics.hpp"
#include "core/lbm.hpp"

#include "problems/homogeneousTurbulence.hpp"

// void reportDevices()
// {
//     int deviceCount {0};
//     cuda_CHECK(cudaGetDeviceCount(&deviceCount));

//     for (int id = 0; id < deviceCount; ++id)
//     {
//         cudaDeviceProp_t prop;
//         cuda_CHECK(cudaGetDeviceProperties(&prop, id));

//         #if defined(__cuda_PLATFORM_AMD__)
//             std::string archName {prop.gcnArchName};
//         #else
//             std::string archName {"sm_" + std::to_string(prop.major) + std::to_string(prop.minor)};
//         #endif

//         std::cout << "Device " << id
//                 << " | " << std::setw(20) << std::left << prop.name
//                 << " | " << std::setw(8) << std::left << archName
//                 << " | " << std::setw(6) << std::right
//                 << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
//     }
// }

constexpr int NX {256};
constexpr int NY {256};
constexpr int NZ {256};

constexpr dfloat TAU {0.51637};
constexpr dfloat OMEGA_EQ {1 / TAU};
constexpr dfloat OMEGA_NEQ {1 - OMEGA_EQ};


static_assert(NX % BLOCK_SIZE_X == 0);
static_assert(NY % BLOCK_SIZE_Y == 0);
static_assert(NZ % BLOCK_SIZE_Z == 0);
static_assert(BLOCK_VOL <= 1024);


constexpr int N_STEPS {389254};


int main()
{
    cuda_CHECK(cudaSetDevice(1));

    constexpr size_t numberOfNodes = static_cast<size_t>(NX) * NY * NZ;
    constexpr size_t numberOfElements = numberOfNodes * NUMBER_OF_MOMENTS;
    dim3 block {BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z};
    dim3 grid {NX / BLOCK_SIZE_X, NY / BLOCK_SIZE_Y, NZ / BLOCK_SIZE_Z};

    dfloat* momentsA = nullptr;
    dfloat* momentsB = nullptr;
    double* dkPartial = nullptr;
    double* depsPartial = nullptr;

    std::vector<double> hkPartial(RED_BLOCKS), hepsPartial(RED_BLOCKS);

    cuda_CHECK(cudaMalloc(&momentsA, numberOfElements * sizeof(dfloat)));
    cuda_CHECK(cudaMalloc(&momentsB, numberOfElements * sizeof(dfloat)));
    cuda_CHECK(cudaMalloc(&dkPartial, RED_BLOCKS * sizeof(double)));
    cuda_CHECK(cudaMalloc(&depsPartial, RED_BLOCKS * sizeof(double)));


    /// LOAD INITIAL MOMENTS FROM VTK FILE
    loadInitialConditions(momentsA, grid, block, NX, NY, NZ, TAU);    
    // ================================================
    

    //setInitialConditions<<<grid, block>>>(momentsA, NX, NY, NZ, TAU);

    cuda_CHECK(cudaGetLastError());
    cuda_CHECK(cudaDeviceSynchronize());

    std::ofstream file("statistics_d3q27.dat");
    file << "#t\tk\teps\n";
    const auto t0 = std::chrono::steady_clock::now();

    int reportInterval = N_STEPS / 100 ;

    reduceStatistics<<<RED_BLOCKS, RED_THREADS>>>(momentsA, dkPartial, depsPartial, numberOfNodes, TAU);
    cuda_CHECK(cudaGetLastError());
    cuda_CHECK(cudaMemcpy(hkPartial.data(), dkPartial, RED_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost));
    cuda_CHECK(cudaMemcpy(hepsPartial.data(), depsPartial, RED_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost));
    const double k   = std::accumulate(hkPartial.begin(),   hkPartial.end(),   0.0);
    const double eps = std::accumulate(hepsPartial.begin(), hepsPartial.end(), 0.0);
    file << std::scientific << std::setprecision(10) << 0 << '\t' << k / numberOfNodes << '\t' << eps / numberOfNodes << '\n';
    file.flush();

    for (int t = 1; t <= N_STEPS+1; ++t)
    {
        kernel<<<grid, block>>>(momentsA, momentsB, NX, NY, NZ, OMEGA_EQ, OMEGA_NEQ);
        cuda_CHECK(cudaGetLastError());
        std::swap(momentsA, momentsB);

        if (t % 500 == 0)
        {
            reduceStatistics<<<RED_BLOCKS, RED_THREADS>>>(momentsA, dkPartial, depsPartial, numberOfNodes, TAU);
            cuda_CHECK(cudaGetLastError());
            cuda_CHECK(cudaMemcpy(hkPartial.data(), dkPartial, RED_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost));
            cuda_CHECK(cudaMemcpy(hepsPartial.data(), depsPartial, RED_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost));
            const double k   = std::accumulate(hkPartial.begin(),   hkPartial.end(),   0.0);
            const double eps = std::accumulate(hepsPartial.begin(), hepsPartial.end(), 0.0);
            file << std::scientific << std::setprecision(10) << t << '\t' << k / numberOfNodes << '\t' << eps / numberOfNodes << '\n';
            file.flush();
        }

        if (t % reportInterval == 0 && t > 0)
        {
            cuda_CHECK(cudaDeviceSynchronize());
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double eta = elapsed * (N_STEPS - t) / t;
            const int h_el = static_cast<int>(elapsed) / 3600;
            const int m_el = (static_cast<int>(elapsed) % 3600) / 60;
            const int h_eta = static_cast<int>(eta) / 3600;
            const int m_eta = (static_cast<int>(eta) % 3600) / 60;
            std::cout << "Step: " << t << '\t' << 100 * t / N_STEPS << "%"
                      << "\tElapsed: " << h_el << "h" << m_el << "min" << "\tETA: "
                      << std::setw(2) << h_eta << "h"
                      << std::setw(2) << m_eta << "min" << std::setfill(' ') << '\n';
        }
    }

    cuda_CHECK(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "MLUPS: " << static_cast<double>(numberOfNodes) * N_STEPS / (sec * 1.0e6) << '\n';

    cuda_CHECK(cudaFree(momentsA));
    cuda_CHECK(cudaFree(momentsB));
    cuda_CHECK(cudaFree(dkPartial));
    cuda_CHECK(cudaFree(depsPartial));

    return 0;
}
