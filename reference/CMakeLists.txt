add_library(GLC_LIB gemm.cpp)

set (CMAKE_CXX_FLAGS "-O3 -std=c++11 -mkl=parallel")

add_executable(gemm_reference main_reference.cpp)

target_link_libraries(gemm_reference GLC_LIB)

add_custom_command(TARGET gemm_reference
				   POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy gemm_reference ${PROJECT_BINARY_DIR}/)

add_subdirectory(test)
