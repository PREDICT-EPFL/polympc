cmake_minimum_required(VERSION 3.0)
project(polympc)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Tests should be Debug build to enable Eigen asserts
set(CMAKE_BUILD_TYPE "Debug")

add_executable(qp_solver_test qp_solver_test.cpp qp_solver_sparse_test.cpp)
target_link_libraries(qp_solver_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})

add_executable(sqp_test sqp_test.cpp)
target_link_libraries(sqp_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})

add_executable(sqp_test_autodiff sqp_test_autodiff.cpp)
target_link_libraries(sqp_test_autodiff Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})

add_executable(bfgs_test bfgs_test.cpp)
target_link_libraries(bfgs_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES})

