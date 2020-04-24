cmake_minimum_required(VERSION 3.5)
project(polympc)

include_directories(include ${CASADI_INCLUDE_DIR})

add_library(polymath polymath.cpp polymath.h)
target_link_libraries(polymath ${CASADI_LIBRARIES} Eigen3::Eigen)


