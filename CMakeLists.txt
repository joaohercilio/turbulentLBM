cmake_minimum_required(VERSION 3.20)

project(hip_test LANGUAGES CUDA)

add_executable(hip_test main.cpp)
target_compile_definitions(hip_test PRIVATE)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(hip_test PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

