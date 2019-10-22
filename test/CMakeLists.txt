set(GOOGLETEST_ROOT external/googletest/googletest CACHE STRING "Google Test source root")

include_directories(
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/include
    )

set(GOOGLETEST_SOURCES
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/src/gtest-all.cc
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/src/gtest_main.cc
    )

foreach(_source ${GOOGLETEST_SOURCES})
    set_source_files_properties(${_source} PROPERTIES GENERATED 1)
endforeach()

add_library(googletest ${GOOGLETEST_SOURCES})

# GEMM OpenACC
add_executable(GEMMOpenACC main.cpp OpenACC/gemm.cpp)

target_link_libraries(GEMMOpenACC googletest GLC_LIB GLC_OACC_LIB pthread)

add_test(NAME GEMMOpenACC COMMAND GEMMOpenACC)

# GEMM OpenMP
add_executable(GEMMOpenMP main.cpp OpenMP/gemm.cpp)

target_link_libraries(GEMMOpenMP googletest GLC_LIB GLC_OMP_LIB pthread)

add_test(NAME GEMMOpenMP COMMAND GEMMOpenMP)

# GEMM reference
add_executable(GEMMReference main.cpp reference/gemm.cpp)

target_link_libraries(GEMMReference googletest GLC_LIB pthread)

add_test(NAME GEMMReference COMMAND GEMMReference)
