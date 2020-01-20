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
set (CMAKE_CXX_FLAGS "-O3 -std=c++11 -acc -Minfo=accel -ta=tesla,cc60 -Mcuda")

add_executable(GEMMOpenACCTest main.cpp OpenACC/gemm.cpp)

target_link_libraries(GEMMOpenACCTest googletest GLC_LIB GLC_OACC_LIB pthread)

add_test(NAME GEMMOpenACCTest COMMAND GEMMOpenACCTest)

# GEMM OpenMP
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")

add_executable(GEMMOpenMPTest main.cpp OpenMP/gemm.cpp)

target_link_libraries(GEMMOpenMPTest googletest GLC_LIB GLC_OMP_LIB pthread)

add_test(NAME GEMMOpenMPTest COMMAND GEMMOpenMPTest)

# GEMM reference
set (CMAKE_CXX_FLAGS "-O3 -std=c++11")

add_executable(GEMMReferenceTest main.cpp reference/gemm.cpp)

target_link_libraries(GEMMReferenceTest googletest GLC_LIB pthread)

add_test(NAME GEMMReferenceTest COMMAND GEMMReferenceTest)
