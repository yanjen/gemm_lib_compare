cmake_minimum_required (VERSION 3.10)

add_library(GLC_OACC_LIB gemm.cpp)

include_directories ("${PROJECT_SOURCE_DIR}/../include")

add_executable(gemm_OpenACC main_OpenACC.cpp)

target_link_libraries(gemm_OpenACC GLC_OACC_LIB)

add_custom_command(TARGET gemm_OpenACC
				   POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy gemm_OpenACC ${PROJECT_BINARY_DIR}/../)
