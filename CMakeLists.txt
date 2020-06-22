cmake_minimum_required(VERSION 3.5)
project(polympc)

## is used, also find other catkin packages
set(CMAKE_MODULE_PATH /usr/local/share/cmake/Modules/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake)


find_package(CASADI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

option(BUILD_TESTS "Build all tests in PolyMPC project" OFF)
option(CONTROL_TESTS "Build collocation tests" OFF)
option(QP_TESTS "Build QP solver tests" OFF)
option(SQP_TESTS "Build SQP solver tests" OFF)
option(POLY_TESTS "Build Polynomials tests" OFF)
option(AUTODIFF_TESTS "Build AutoDiff tests" OFF)
option(BUILD_RELEASE "Build release version w/o debugging symbols" OFF)
option(BUILD_EXAMPLES "Build examples for CasADi and Eigen interfaces" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_VERBOSE_MAKEFILE “ON”)
if(BUILD_RELEASE)
  set(CMAKE_BUILD_TYPE "Release")
else()
  set(CMAKE_BUILD_TYPE "Debug")
endif()

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
add_subdirectory(src/integration)

add_subdirectory(tests/control)
add_subdirectory(tests/polynomials)
add_subdirectory(tests/autodiff)
add_subdirectory(tests/solvers/qp)
add_subdirectory(tests/solvers/sqp)
