cmake_minimum_required(VERSION 2.8.3)
project(polynomials)

enable_testing()

add_executable(auto_diff_test auto_diff_test.cpp)
target_link_libraries(auto_diff_test ${CASADI_LIBRARIES} Eigen3::Eigen)

add_executable(auto_diff_test2 auto_diff_test2.cpp)
target_link_libraries(auto_diff_test2 Eigen3::Eigen)
