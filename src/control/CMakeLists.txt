cmake_minimum_required(VERSION 2.8.3)
project(control)

include_directories(include ${CASADI_INCLUDE_DIR})

add_executable(dynamics_approx_test dynamics_approx_test.cpp)
target_link_libraries(dynamics_approx_test Eigen3::Eigen)

add_executable(cost_approx_test cost_approx_test.cpp)
target_link_libraries(cost_approx_test Eigen3::Eigen)
