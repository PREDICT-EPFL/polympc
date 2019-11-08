cmake_minimum_required(VERSION 2.8.3)
project(qp)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(qp_solver_test qp_solver_test.cpp qp_solver_sparse_test.cpp test_main.cpp)
target_link_libraries(qp_solver_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES} pthread)
