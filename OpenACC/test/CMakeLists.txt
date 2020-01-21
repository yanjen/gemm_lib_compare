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

# GEMM OpenACC
set (CMAKE_CXX_FLAGS "-O3 -std=c++11 -acc -Minfo=accel -ta=tesla,cc60 -Mcuda")

add_library(GLC_LIB STATIC IMPORTED)

set_target_properties(GLC_LIB PROPERTIES
	IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/../reference/libGLC_LIB.a)

add_executable(GEMMOpenACCTest main.cpp gemm.cpp)

target_link_libraries(GEMMOpenACCTest googletest GLC_LIB GLC_OACC_LIB pthread)

add_custom_command(TARGET GEMMOpenACCTest
				   POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy GEMMOpenACCTest ${PROJECT_BINARY_DIR}/..)
