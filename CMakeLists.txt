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
    set(CASADI_FOUND False)
endif()

option(BUILD_TESTS "Build all tests in PolyMPC project" OFF)
option(BUILD_RELEASE "Build release version w/o debugging symbols" ON)
option(BUILD_EXAMPLES "Build examples for CasADi and Eigen interfaces" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_VERBOSE_MAKEFILE “ON”)


include_directories(src src/examples ${CASADI_INCLUDE_DIR} ${Ipopt_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

add_subdirectory(src cmake)
add_subdirectory(examples)
add_subdirectory(src/control)
add_subdirectory(src/solvers)
add_subdirectory(src/integration)

add_subdirectory(tests)
add_subdirectory(tests/polynomials)
add_subdirectory(tests/autodiff)
