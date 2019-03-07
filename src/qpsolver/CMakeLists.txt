cmake_minimum_required(VERSION 3.0)
project(qpsolver)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(osqp_test osqp_test.cpp)
target_link_libraries(osqp_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})
