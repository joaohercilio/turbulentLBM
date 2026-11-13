cmake_minimum_required(VERSION 3.20)

project(mTurbulentLBM LANGUAGES C CUDA)

set(CMAKE_CUDA_STANDARD 20)

find_package(HDF5 REQUIRED COMPONENTS C)

add_executable(mTurbulentLBM main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(mTurbulentLBM PRIVATE HDF5::HDF5)
