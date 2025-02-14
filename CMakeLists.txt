cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
cmake_policy(SET CMP0112 NEW)

project(GNN  LANGUAGES CXX CUDA)

set(CUTLASS_NVCC_ARCHS "80")

set(CUTLASS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)

file(GLOB_RECURSE CUTLASS_INCLUDE RELATIVE ${CUTLASS_PATH} include/cutlass/*.h)
file(GLOB_RECURSE CUTLASS_CUTLASS RELATIVE ${CUTLASS_PATH}/include include/cutlass/*.h include/cutlass/*.hpp include/cutlass/*.inl)
file(GLOB_RECURSE CUTLASS_CUTE RELATIVE ${CUTLASS_PATH}/include include/cute/*.h*)
# file(GLOB_RECURSE CUTLASS_NVRTC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/test test/unit/nvrtc/kernel/*.h)
include_directories(${CUTLASS_PATH}/include)
add_executable(basic src/basic_gemm.cu)