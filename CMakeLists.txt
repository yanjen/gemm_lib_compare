cmake_minimum_required (VERSION 3.10)
project (GPU_lib_compare)
# The version number.
set (GLC_VERSION_MAJOR 0)
set (GLC_VERSION_MINOR 0)
set (GLC_VERSION_PATCH 0)

include (ExternalProject)

set(Openblas_DIR "${Openblas_DIR}" CACHE PATH "Path to OpenBLAS.")

include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories ("${Openblas_DIR}/include")

set (CMAKE_CXX_COMPILER icpc)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

add_subdirectory(reference)

ExternalProject_Add(GEMMOpenMP
    PREFIX ${PROJECT_BINARY_DIR}/OpenMP
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/OpenMP
    BINARY_DIR ${PROJECT_BINARY_DIR}/OpenMP
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_CXX_COMPILER=icpc
        "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -qopenmp"
    INSTALL_COMMAND ""
)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/OpenMP)

ExternalProject_Add(GEMMOpenACC
    PREFIX ${PROJECT_BINARY_DIR}/OpenACC
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/OpenACC
    BINARY_DIR ${PROJECT_BINARY_DIR}/OpenACC
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_CXX_COMPILER=pgc++
        "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -acc -Minfo=accel -ta=tesla,cc60 -Mcuda"
    INSTALL_COMMAND ""
)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/OpenACC)
