cmake_minimum_required(VERSION 2.8.3)
project(examples)

include_directories(include ${CASADI_INCLUDE_DIR} ${Eigen3_INCLUDE_DIR})

add_library(kite kite.cpp kite.h)
target_link_libraries(kite polymath ${CASADI_LIBRARIES} Eigen3::Eigen)

add_library(mobile_robot mobile_robot.cpp mobile_robot.hpp)
target_link_libraries(mobile_robot ${CASADI_LIBRARIES})

add_executable(robot_control_test robot_control_test.cpp)
target_link_libraries(robot_control_test mobile_robot)

add_executable(kite_model_test kite_model_test.cpp)
target_link_libraries(kite_model_test kite odesolver Eigen3::Eigen)

add_executable(ebyshev_test ebyshev_test.cpp)
target_link_libraries(ebyshev_test polymath Eigen3::Eigen)

add_executable(chebyshev_test chebyshev_test.cpp)
target_link_libraries(chebyshev_test polymath Eigen3::Eigen)

add_executable(legendre_test legendre_test.cpp)
target_link_libraries(legendre_test polymath Eigen3::Eigen)

add_executable(kite_sode_test kite_sode_test.cpp)
target_link_libraries(kite_sode_test polymath Eigen3::Eigen)

add_executable(kite_control_test kite_control_test.cpp)
target_link_libraries(kite_control_test kite Eigen3::Eigen)

add_executable(auto_diff_test auto_diff_test.cpp)
target_link_libraries(auto_diff_test ${CASADI_LIBRARIES} Eigen3::Eigen)

add_executable(auto_diff_test2 auto_diff_test2.cpp)
target_link_libraries(auto_diff_test2 Eigen3::Eigen)
