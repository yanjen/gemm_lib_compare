set(GOOGLETEST_ROOT external/googletest/googletest CACHE STRING "Google Test source root")

include_directories(
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/include
    )

# GEMM reference
set (CMAKE_CXX_FLAGS "-O3 -std=c++11")

add_executable(GEMMReferenceTest main.cpp gemm.cpp)

target_link_libraries(GEMMReferenceTest googletest GLC_LIB pthread)

add_custom_command(TARGET GEMMReferenceTest
				   POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy GEMMReferenceTest ${PROJECT_BINARY_DIR}/)
