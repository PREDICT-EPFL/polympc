cmake_minimum_required(VERSION 3.5)

if(BUILD_TESTS OR QP_TESTS)

  enable_testing()

  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})

  add_executable(qp_solver_test admm_solver_test.cpp
                                box_admm_test.cpp
                                qpmad_interface_test.cpp
                                qp_solver_sparse_test.cpp
                                test_main.cpp)
  target_link_libraries(qp_solver_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES} pthread)

  if(OsqpEigen_FOUND)
      target_link_libraries(qp_solver_test OsqpEigen::OsqpEigen)
  endif()

  #add_executable(qp_base_test qp_base_test.cpp)
  #target_link_libraries(qp_base_test Eigen3::Eigen)

endif()
