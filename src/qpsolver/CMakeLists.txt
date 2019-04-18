cmake_minimum_required(VERSION 3.0)
project(qpsolver)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(osqp_test osqp_test.cpp)
target_link_libraries(osqp_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})

add_executable(sqp_test sqp_test.cpp)
target_link_libraries(sqp_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})

add_executable(sqp_test_autodiff sqp_test_autodiff.cpp)
target_link_libraries(sqp_test_autodiff Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})

add_executable(bfgs_test bfgs_test.cpp)
target_link_libraries(bfgs_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})

