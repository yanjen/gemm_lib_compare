add_library(GLC_LIB gemm.cpp)

link_directories("${Openblas_DIR}/lib")

add_executable(gemm_reference main_reference.cpp)

target_link_libraries(gemm_reference GLC_LIB openblas)

add_custom_command(TARGET gemm_reference
				   POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy gemm_reference ${PROJECT_BINARY_DIR}/)

add_subdirectory(test)
