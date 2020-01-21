set(GOOGLETEST_ROOT ../external/googletest/googletest CACHE STRING "Google Test source root")

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

# GEMM OpenMP
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")

add_library(GLC_LIB STATIC IMPORTED)

set_target_properties(GLC_LIB PROPERTIES
	IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/../reference/libGLC_LIB.a)

add_executable(GEMMOpenMPTest main.cpp gemm.cpp)

target_link_libraries(GEMMOpenMPTest googletest GLC_LIB GLC_OMP_LIB pthread)

add_custom_command(TARGET GEMMOpenMPTest
				   POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy GEMMOpenMPTest ${PROJECT_BINARY_DIR}/..)
