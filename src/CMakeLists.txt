cmake_minimum_required(VERSION 3.5)
project(polympc)

if(${CASADI_FOUND})
    add_library(polymath polymath.cpp polymath.h)
    target_link_libraries(polymath ${CASADI_LIBRARIES} Eigen3::Eigen)
endif()


