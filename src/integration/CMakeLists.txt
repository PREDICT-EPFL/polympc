cmake_minimum_required(VERSION 3.5)
project(polympc)

include_directories(include ${CASADI_INCLUDE_DIR})

add_library(integration integrator.cpp integrator.h)
target_link_libraries(integration ${CASADI_LIBRARIES} Eigen3::Eigen)
