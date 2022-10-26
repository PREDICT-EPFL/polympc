cmake_minimum_required(VERSION 3.5)
project(polympc VERSION 0.1.0)

## is used, also find other catkin packages
set(CMAKE_MODULE_PATH /usr/local/share/cmake/Modules/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake)
set( ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig" )


find_package(CASADI)
find_package(QPMAD)
find_package(Ipopt)
find_package(OsqpEigen)
find_package(Eigen3 REQUIRED NO_MODULE)

if(Ipopt_FOUND)
    message("Found Ipopt [x]")
else()
    message("Ipopt is not found, the corresponding interface and examples will not be built")
    set(Ipopt_INCLUDE_DIRS "")
endif()

if(${CASADI_FOUND})
    message("Found CasADi [x]")
else()
    message(WARNING "CasADi is not found, the corresponding interface and examples will not be built")
    set(CASADI_INCLUDE_DIR "")
endif()

option(BUILD_TESTS "Build all tests in PolyMPC project" OFF)
option(CONTROL_TESTS "Build collocation tests" OFF)
option(QP_TESTS "Build QP solver tests" OFF)
option(SQP_TESTS "Build SQP solver tests" OFF)
option(POLY_TESTS "Build Polynomials tests" OFF)
option(AUTODIFF_TESTS "Build AutoDiff tests" OFF)
option(BUILD_RELEASE "Build release version w/o debugging symbols" ON)
option(BUILD_EXAMPLES "Build examples for CasADi and Eigen interfaces" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_VERBOSE_MAKEFILE “ON”)

if(BUILD_RELEASE)
  set(CMAKE_BUILD_TYPE "Release")
else()
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if(QPMAD_FOUND)
    add_compile_definitions(POLYMPC_FOUND_QPMAD=1)
endif()

if(OsqpEigen_FOUND)
    add_compile_definitions(POLYMPC_FOUND_OSQP_EIGEN=1)
endif()

#add_definitions(-DEIGEN_DONT_VECTORIZE=1)
#add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1000000) # 1 MB


include_directories(src src/examples ${CASADI_INCLUDE_DIR} ${Ipopt_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

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
add_subdirectory(tests/solvers/ipopt)
