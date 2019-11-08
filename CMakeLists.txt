cmake_minimum_required(VERSION 2.8.3)
project(polympc)

## is used, also find other catkin packages
set(CMAKE_MODULE_PATH /usr/local/share/cmake/Modules/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake)


find_package(CASADI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_VERBOSE_MAKEFILE “ON”)
set(CMAKE_BUILD_TYPE "Debug")

#detect 32bit system
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "Target architecture 64 bits")
else()
    message(STATUS "Target architecture 32 bits")
endif()


#if(ARM_ENABLE_VECTORIZATION AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" )
#    message(STATUS "EIGEN: ARM VECTORIZATION ENABLED")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
#endif()


include_directories(src src/examples ${CASADI_INCLUDE_DIR} )

add_subdirectory(src cmake)
add_subdirectory(examples)
add_subdirectory(src/control)
add_subdirectory(src/solvers)

add_subdirectory(tests/control)
add_subdirectory(tests/polynomials)
add_subdirectory(tests/autodiff)
add_subdirectory(tests/solvers/qp)
add_subdirectory(tests/solvers/sqp)
