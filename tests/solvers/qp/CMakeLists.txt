cmake_minimum_required(VERSION 3.5)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

if(BUILD_TESTS OR QP_TESTS)

  add_executable(qp_solver_test qp_solver_test.cpp qp_solver_sparse_test.cpp test_main.cpp)
  target_link_libraries(qp_solver_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(qp_test_1D qp_solver_tests_1D.cpp test_main.cpp)
  target_link_libraries(qp_test_1D Eigen3::Eigen ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(qp_test_qplib qp_solver_tests_qplib.cpp test_main.cpp load_matrix_from_csv.hpp)
  target_link_libraries(qp_test_qplib Eigen3::Eigen ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(qp_base_test qp_base_test.cpp)
  target_link_libraries(qp_base_test Eigen3::Eigen)

endif()
