cmake_minimum_required (VERSION 3.10)
project (GPU_lib_compare)
# The version number.
set (GLC_VERSION_MAJOR 0)
set (GLC_VERSION_MINOR 0)
set (GLC_VERSION_PATCH 0)

include_directories ("${PROJECT_SOURCE_DIR}/include")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl")

add_subdirectory(reference)

add_executable(gemm main.cpp)

target_link_libraries(gemm GLC_LIB)

include (CTest)
add_subdirectory(test)
