# GPU_lib_compare
This repository compares the runtime performance of CUDA, OpenACC, and OpenMP on GPU with several applications.

## Getting Started

### Prerequisities
The following packages are required to compile this project.

- __PGI C++ compiler__ (>18.1)
- __NVIDIA CUDA compiler__
- __CMake__ (>3.10)

__Note that the project has to be run on a mechine with NVIDIA GPU__

__(The default compiler flag option is set for NVIDIA Tesla P100)__

### Compiling
Run the following command to compile the project
```
mkdir build
cd build
cmake ..
make
```

## Running the tests
As the executable files are compiled with multiple compilers, we separate the tests into multiple executable files as well. To test the gemm accuracy, run the following tests after compiling.
```
./GEMMReferenceTest
./GEMMOpenMPTest
./GEMMOpenACCTest
```
Seperate tests are under the directory `test/`.

## Running speed compare
After Compiling, run the following command.
```
./gemm_reference
./gemm_OpenMP
./gemm_OpenACC
```
