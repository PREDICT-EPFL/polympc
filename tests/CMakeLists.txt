cmake_minimum_required(VERSION 3.10)

if(BUILD_TESTS)

  include(FetchContent)
  FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  ## QP SOLVERS TESTS ##

  # ADMM and boxADMM tests
  add_executable(admm_solver_test solvers/qp/admm_solver_test.cpp
                                  solvers/qp/box_admm_test.cpp
                                  solvers/qp/qp_solver_sparse_test.cpp
                                  solvers/qp/test_main.cpp)

  target_link_libraries(admm_solver_test Eigen3::Eigen GTest::gtest_main GTest::gtest)

  include(GoogleTest)
  gtest_discover_tests(admm_solver_test)

  # OSQP interface test
  if(OsqpEigen_FOUND)
      add_executable(osqp_solver_test solvers/qp/osqp_solver_test.cpp)
      target_link_libraries(osqp_solver_test Eigen3::Eigen OsqpEigen::OsqpEigen GTest::gtest_main GTest::gtest)
      gtest_discover_tests(osqp_solver_test)
  endif()

  # QPMAD interface test
  if(QPMAD_FOUND)
      add_executable(qpmad_solver_test solvers/qp/qpmad_interface_test.cpp)
      target_link_libraries(qpmad_solver_test Eigen3::Eigen GTest::gtest_main GTest::gtest)
      gtest_discover_tests(qpmad_solver_test)
  endif()

  ## NONLINEAR SOLVERS TESTS ##

  # SQP solver tests
  add_executable(sqp_solver_test solvers/sqp/sqp_test_autodiff.cpp
                                 solvers/sqp/test_main.cpp)
  target_link_libraries(sqp_solver_test Eigen3::Eigen GTest::gtest_main GTest::gtest)
  gtest_discover_tests(sqp_solver_test)

  # Codegen SQP test
  file(GLOB CASADI_CODEGEN "solvers/sqp/casadi_codegen/*.cpp" "solvers/sqp/casadi_codegen/*.h")
  add_executable(sqp_codegen_test solvers/sqp/codegen_test.cpp ${CASADI_CODEGEN})
  target_link_libraries(sqp_codegen_test Eigen3::Eigen GTest::gtest_main GTest::gtest)
  gtest_discover_tests(sqp_codegen_test)

  # BFGS-update test
  add_executable(bfgs_test solvers/sqp/bfgs_test.cpp)
  target_link_libraries(bfgs_test Eigen3::Eigen GTest::gtest_main GTest::gtest)
  gtest_discover_tests(bfgs_test)

  # IPOPT interface test
  if(Ipopt_FOUND)
      add_executable(ipopt_interface_test solvers/ipopt/ipopt_interface_test.cpp)
      target_link_libraries(ipopt_interface_test Eigen3::Eigen ${Ipopt_LIBRARIES} GTest::gtest_main GTest::gtest)
      gtest_discover_tests(ipopt_interface_test)
  endif()

  ## CONTROL TESTS ##

  # CSTR stabilisation NMPC test
  add_executable(cstr_control_test control/cstr_control_test.cpp)
  target_link_libraries(cstr_control_test Eigen3::Eigen GTest::gtest_main GTest::gtest)
  gtest_discover_tests(cstr_control_test)

  # Minimum-time valet parking test
  add_executable(minimal_time_test control/minimal_time_test.cpp)
  target_link_libraries(minimal_time_test Eigen3::Eigen GTest::gtest_main GTest::gtest)
  gtest_discover_tests(minimal_time_test)

  # MPC wrapper test
  add_executable(mpc_wrapper_test control/mpc_wrapper_test.cpp)
  target_link_libraries(mpc_wrapper_test Eigen3::Eigen GTest::gtest_main GTest::gtest)
  gtest_discover_tests(mpc_wrapper_test)

  # Compare dense and sparse collocation of the minimum-time valet parking problem
  add_executable(dense_sparse_compare control/dense_sparse_compare.cpp)
  target_link_libraries(dense_sparse_compare Eigen3::Eigen GTest::gtest_main GTest::gtest)
  gtest_discover_tests(dense_sparse_compare)

  add_executable(valet_parking_test control/valet_parking_mpc_test.cpp)
  target_link_libraries(valet_parking_test Eigen3::Eigen GTest::gtest_main GTest::gtest)
  gtest_discover_tests(valet_parking_test)

#  add_executable(continuous_ocp_test continuous_ocp_test.cpp)
#  target_link_libraries(continuous_ocp_test Eigen3::Eigen)

#  add_executable(lqr_test lqr_test.cpp)
#  target_link_libraries(lqr_test Eigen3::Eigen)

endif()
