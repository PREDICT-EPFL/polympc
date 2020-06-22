cmake_minimum_required(VERSION 3.5)
project(integration)

add_library(integration integrator.cpp integrator.h)
target_link_libraries(integration ${CASADI_LIBRARIES} Eigen3::Eigen)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
