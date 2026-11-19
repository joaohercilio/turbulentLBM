#!/bin/bash
cd "$(dirname "$0")/../.."

build() {
    B=build/$1_x$2_gpu$3
    rm -rf $B
    mkdir -p $B/src
    cp -r CMakeLists.txt main.cpp core lattice problems $B/src/
    sed -i "s/using namespace d3q27_2nd;/using namespace $1;/" $B/src/lattice/lattice.hpp
    sed -i "s/cudaSetDevice(1)/cudaSetDevice($3)/" $B/src/main.cpp
    sed -i "s/constexpr double U_SCALE {1};/constexpr double U_SCALE {$2};/" $B/src/problems/homogeneousTurbulence.hpp
    cmake -S $B/src -B $B > /dev/null && make -C $B > /dev/null || echo "falhou: $B"
}
export -f build

rm -rf build
for ns in d3q15_2nd d3q19_2nd d3q27_2nd d3q15_3rd d3q19_3rd d3q27_3rd; do
    for b in 1 2 3; do
        for g in 0 1; do echo $ns $b $g; done
    done
done | xargs -P 8 -n 3 bash -c 'build "$@"' _
