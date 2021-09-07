cmake_minimum_required(VERSION 3.5)

if(BUILD_TESTS AND Ipopt_FOUND)

  enable_testing()

  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})

  add_executable(ipopt_test ipopt_interface_test.cpp test_main.cpp)
  target_link_libraries(ipopt_test Eigen3::Eigen ${Ipopt_LIBRARIES} ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(ipopt_ocp_test ipopt_ocp_test.cpp)
  target_link_libraries(ipopt_ocp_test Eigen3::Eigen ${Ipopt_LIBRARIES} ${GTEST_BOTH_LIBRARIES} pthread)

endif()
