cmake_minimum_required(VERSION 3.5)

if(BUILD_TESTS OR SQP_TESTS)

  enable_testing()

  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})

  #add_executable(sqp_test sqp_test.cpp sqp_test_autodiff.cpp bfgs_test.cpp test_main.cpp)
  #target_link_libraries(sqp_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(sqp_test sqp_test_autodiff.cpp test_main.cpp)
  target_link_libraries(sqp_test Eigen3::Eigen ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(sqp_base_test sqp_base_test.cpp)
  target_link_libraries(sqp_base_test Eigen3::Eigen)

  if(OsqpEigen_FOUND)
      target_link_libraries(sqp_base_test OsqpEigen::OsqpEigen)
  endif()

  #codegen test
  file(GLOB CASADI_CODEGEN "casadi_codegen/*.cpp" "casadi_codegen/*.h")
  add_executable(codegen_test codegen_test.cpp ${CASADI_CODEGEN})
  target_link_libraries(codegen_test Eigen3::Eigen)


endif()
