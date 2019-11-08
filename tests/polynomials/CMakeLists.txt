cmake_minimum_required(VERSION 2.8.3)
project(polynomials)

enable_testing()

add_executable(legendre_test legendre_test.cpp)
target_link_libraries(legendre_test polymath Eigen3::Eigen)

add_executable(ebyshev_test ebyshev_test.cpp)
target_link_libraries(ebyshev_test polymath Eigen3::Eigen)
