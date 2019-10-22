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
cmake -D CMAKE_C_COMPILER="/path/to/your/pgi/compiler/pgc++" ..
make
```

## Running the tests
After Compiling, run the following command.
```
make test
```
Seperate tests are under the directory `test/`.

## Running speed compare
After Compiling, run the following command.
```
./gemm
```
