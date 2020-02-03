enable_language(CUDA)

set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-gpu-targets")
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60 -D_FORCE_INLINES")

add_library(GLC_CUDA_LIB gemm.cu)

add_executable(gemm_CUDA main_CUDA.cu)

target_link_libraries(gemm_CUDA GLC_CUDA_LIB)

add_custom_command(TARGET gemm_CUDA
				   POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy gemm_CUDA ${PROJECT_BINARY_DIR}/)
