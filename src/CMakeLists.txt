cmake_minimum_required(VERSION 2.8.3)
project(polympc)

include_directories(include ${CASADI_INCLUDE_DIR})

add_library(polymath polymath.cpp polymath.h)
target_link_libraries(polymath ${CASADI_LIBRARIES} Eigen3::Eigen)

add_library(odesolver integrator.cpp integrator.h)
target_link_libraries(odesolver polymath ${CASADI_LIBRARIES} Eigen3::Eigen)

add_executable(casadi_test casadi_test.cpp)
target_link_libraries(casadi_test ${CASADI_LIBRARIES} Eigen3::Eigen)

