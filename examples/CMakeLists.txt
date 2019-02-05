cmake_minimum_required(VERSION 2.8.3)
project(examples)

include_directories(include ${CASADI_INCLUDE_DIR})

add_library(kite kite.cpp kite.h)
target_link_libraries(kite polymath ${CASADI_LIBRARIES})

add_executable(kite_model_test kite_model_test.cpp)
target_link_libraries(kite_model_test kite odesolver)

add_executable(ebyshev_test ebyshev_test.cpp)
target_link_libraries(ebyshev_test polymath)

add_executable(chebyshev_test chebyshev_test.cpp)
target_link_libraries(chebyshev_test polymath)

add_executable(legendre_test legendre_test.cpp)
target_link_libraries(legendre_test polymath)

add_executable(kite_sode_test kite_sode_test.cpp)
target_link_libraries(kite_sode_test polymath)

add_executable(kite_control_test kite_control_test.cpp)
target_link_libraries(kite_control_test kite)

add_executable(auto_diff_test auto_diff_test.cpp)
target_link_libraries(auto_diff_test ${CASADI_LIBRARIES})

add_executable(auto_diff_test2 auto_diff_test2.cpp)
#target_link_libraries(auto_diff_test ${CASADI_LIBRARIES})
