cmake_minimum_required (VERSION 3.10)

add_library(GLC_OMP_LIB gemm.cpp)

include_directories ("${PROJECT_SOURCE_DIR}/../include")

add_executable(gemm_OpenMP main_OpenMP.cpp)

target_link_libraries(gemm_OpenMP GLC_OMP_LIB)

add_custom_command(TARGET gemm_OpenMP
				   POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy gemm_OpenMP ${PROJECT_BINARY_DIR}/../)

add_subdirectory(test)
