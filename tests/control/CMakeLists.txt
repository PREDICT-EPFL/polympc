cmake_minimum_required(VERSION 3.5)

if(BUILD_TESTS OR CONTROL_TESTS)

  enable_testing()

  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})

  add_executable(dynamics_approx_test dynamics_approx_test.cpp)
  target_link_libraries(dynamics_approx_test Eigen3::Eigen)

  add_executable(dynamics_sparse_approx_test dynamics_sparse_approx_test.cpp)
  target_link_libraries(dynamics_sparse_approx_test Eigen3::Eigen)

  add_executable(cost_approx_test cost_approx_test.cpp)
  target_link_libraries(cost_approx_test Eigen3::Eigen)

  add_executable(constraint_collocation_test constraint_collocation_test.cpp)
  target_link_libraries(constraint_collocation_test Eigen3::Eigen)

  add_executable(continuous_ocp_test continuous_ocp_test.cpp)
  target_link_libraries(continuous_ocp_test Eigen3::Eigen)

  add_executable(cstr_control_test_eigen cstr_control_test.cpp)
  target_link_libraries(cstr_control_test_eigen Eigen3::Eigen)

  add_executable(mpc_test mpc_wrapper_test.cpp)
  target_link_libraries(mpc_test Eigen3::Eigen)

  add_executable(minimal_time_test minimal_time_test.cpp)
  target_link_libraries(minimal_time_test Eigen3::Eigen)

  #add_executable(mint_compare mint_compare.cpp)
  #target_link_libraries(mint_compare Eigen3::Eigen)

  add_executable(constraints_test nonlinear_constraints_test.cpp)
  target_link_libraries(constraints_test Eigen3::Eigen)

  add_executable(lqr_test lqr_test.cpp)
  target_link_libraries(lqr_test Eigen3::Eigen)

endif()
