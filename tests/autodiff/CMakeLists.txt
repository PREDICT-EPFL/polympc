cmake_minimum_required(VERSION 3.5)

if(BUILD_TESTS OR AUTODIFF_TESTS)

  enable_testing()

if(${CASADI_FOUND})
  add_executable(auto_diff_test auto_diff_test.cpp)
  target_link_libraries(auto_diff_test ${CASADI_LIBRARIES} Eigen3::Eigen)
endif()

  add_executable(auto_diff_test2 auto_diff_test2.cpp)
  target_link_libraries(auto_diff_test2 Eigen3::Eigen)

  add_executable(auto_diff_math math_func_test.cpp)
  target_link_libraries(auto_diff_math Eigen3::Eigen)

  add_executable(auto_diff_jacobian auto_diff_jacobian.cpp)
  target_link_libraries(auto_diff_jacobian Eigen3::Eigen)

endif()
