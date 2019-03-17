cmake_minimum_required(VERSION 2.8.3)
project(control)

enable_testing()

include_directories(include ${CASADI_INCLUDE_DIR})

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(ocproblem_test ocproblem_test.cpp)
target_link_libraries(ocproblem_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})

add_executable(nmpc_test nmpc_test.cpp)
target_link_libraries(nmpc_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})

add_executable(dynamics_approx_test dynamics_approx_test.cpp)
target_link_libraries(dynamics_approx_test Eigen3::Eigen)

add_executable(cost_approx_test cost_approx_test.cpp)
target_link_libraries(cost_approx_test Eigen3::Eigen)
