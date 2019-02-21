cmake_minimum_required(VERSION 2.8.3)
project(control)

include_directories(include ${CASADI_INCLUDE_DIR})

add_executable(dynamics_approx_test dynamics_approx_test.cpp)

add_executable(cost_approx_test cost_approx_test.cpp)
