cmake_minimum_required(VERSION 3.5)

if(BUILD_TESTS)

  if(${CASADI_FOUND})
    add_executable(legendre_test legendre_test.cpp)
    target_link_libraries(legendre_test polymath Eigen3::Eigen)
  endif()

  add_executable(ebyshev_test ebyshev_test.cpp)
  target_link_libraries(ebyshev_test Eigen3::Eigen)

  add_executable(splines_test splines_test.cpp)
  target_link_libraries(splines_test Eigen3::Eigen)

endif()
