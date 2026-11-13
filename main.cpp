#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>

#include "core/definitions.hpp"
#include "core/statistics.hpp"
#include "core/lbm.hpp"
#include "core/output.hpp"

#include "problems/shearLayer.hpp"

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

const dfloat OMEGA_EQ {1 / TAU};
const dfloat OMEGA_NEQ {1 - OMEGA_EQ};


static_assert(NX % BLOCK_SIZE_X == 0);
static_assert(NY % BLOCK_SIZE_Y == 0);
static_assert(NZ % BLOCK_SIZE_Z == 0);
static_assert(BLOCK_VOL <= 1024);


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


    initialize(momentsA, grid, block);

    cuda_CHECK(cudaGetLastError());
    cuda_CHECK(cudaDeviceSynchronize());

    std::ofstream file("statistics_lbm.dat");
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

    int iout = 0;
    double nextOut = DT_OUT;
    output(momentsA, iout++, 0);

    for (int t = 1; t <= N_STEPS+1; ++t)
    {
        kernel<<<grid, block>>>(momentsA, momentsB, NX, NY, NZ, OMEGA_EQ, OMEGA_NEQ);
        cuda_CHECK(cudaGetLastError());
        std::swap(momentsA, momentsB);
        boundary(momentsA);

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

        if (t == std::lround(nextOut))
        {
            output(momentsA, iout++, t);
            nextOut += DT_OUT;
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
            std::cout << "Step: " << t << '\t'
                    << std::fixed << std::setprecision(1) << 100.0 * t / N_STEPS << "%"
                    << "\tElapsed: " << h_el << "h"
                    << std::setfill('0') << std::setw(2) << m_el << "min"
                    << "\tETA: " << h_eta << "h"
                    << std::setw(2) << m_eta << "min"
                    << std::setfill(' ') << std::defaultfloat
                    << "   \r" << std::flush;        
            }
    }

    cuda_CHECK(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nMLUPS: " << std::fixed << std::setprecision(1) << static_cast<double>(numberOfNodes) * N_STEPS / (sec * 1.0e6) << '\n';

    cuda_CHECK(cudaFree(momentsA));
    cuda_CHECK(cudaFree(momentsB));
    cuda_CHECK(cudaFree(dkPartial));
    cuda_CHECK(cudaFree(depsPartial));

    return 0;
}
