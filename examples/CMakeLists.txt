cmake_minimum_required(VERSION 2.8.3)
project(examples)

include_directories(include ${CASADI_INCLUDE_DIR})

add_library(kite kite.cpp kite.h)
target_link_libraries(kite polymath ${CASADI_LIBRARIES})

add_executable(kite_model_test kite_model_test.cpp)
target_link_libraries(kite_model_test kite odesolver)

add_executable(kite_control_test kite_control_test.cpp)
target_link_libraries(kite_control_test kite)
