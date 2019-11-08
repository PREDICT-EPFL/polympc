cmake_minimum_required(VERSION 2.8.3)
project(qp)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(sqp_test sqp_test.cpp sqp_test_autodiff.cpp bfgs_test.cpp test_main.cpp)
target_link_libraries(sqp_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES} pthread)
