cmake_minimum_required(VERSION 3.5)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

if(BUILD_TESTS OR CONTROL_TESTS)

  add_executable(ocproblem_test ocproblem_test.cpp)
  target_link_libraries(ocproblem_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(nmpc_test nmpc_test.cpp)
  target_link_libraries(nmpc_test Eigen3::Eigen)

  add_executable(dynamics_approx_test dynamics_approx_test.cpp)
  target_link_libraries(dynamics_approx_test Eigen3::Eigen)

  add_executable(dynamics_sparse_approx_test dynamics_sparse_approx_test.cpp)
  target_link_libraries(dynamics_sparse_approx_test Eigen3::Eigen)

  add_executable(cost_approx_test cost_approx_test.cpp)
  target_link_libraries(cost_approx_test Eigen3::Eigen)

  add_executable(constraint_collocation_test constraint_collocation_test.cpp)
  target_link_libraries(constraint_collocation_test Eigen3::Eigen)

endif()
