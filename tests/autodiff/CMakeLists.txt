cmake_minimum_required(VERSION 3.5)

enable_testing()

if(BUILD_TESTS OR AUTODIFF_TESTS)

  add_executable(auto_diff_test auto_diff_test.cpp)
  target_link_libraries(auto_diff_test ${CASADI_LIBRARIES} Eigen3::Eigen)

  add_executable(auto_diff_test2 auto_diff_test2.cpp)
  target_link_libraries(auto_diff_test2 Eigen3::Eigen)

endif()
