cmake_minimum_required(VERSION 3.5)

if(${CASADI_FOUND})
	include_directories(include ${CASADI_INCLUDE_DIR} ${Eigen3_INCLUDE_DIR})
else()
	include_directories(include ${Eigen3_INCLUDE_DIR})
endif()

if(BUILD_EXAMPLES AND ${CASADI_FOUND})

    add_library(kite kite.cpp kite.h)
    target_link_libraries(kite polymath ${CASADI_LIBRARIES} Eigen3::Eigen)

    add_library(mobile_robot mobile_robot.cpp mobile_robot.hpp)
    target_link_libraries(mobile_robot ${CASADI_LIBRARIES})

    add_executable(mobile_robot_test mobile_robot_test.cpp)
    target_link_libraries(mobile_robot_test ${CASADI_LIBRARIES})

    add_executable(robot_control_test robot_control_test.cpp)
    target_link_libraries(robot_control_test mobile_robot Eigen3::Eigen)

    add_executable(kite_model_test kite_model_test.cpp)
    target_link_libraries(kite_model_test kite integration Eigen3::Eigen)

    add_executable(car_model_test car_model_test.cpp)
    target_link_libraries(car_model_test integration polymath Eigen3::Eigen)

    add_executable(chebyshev_test chebyshev_test.cpp)
    target_link_libraries(chebyshev_test polymath Eigen3::Eigen)

    add_executable(soft_chebyshev_test soft_chebyshev_test.cpp)
    target_link_libraries(soft_chebyshev_test mobile_robot Eigen3::Eigen)

    add_executable(kite_sode_test kite_sode_test.cpp)
    target_link_libraries(kite_sode_test polymath Eigen3::Eigen)

    add_executable(kite_control_test kite_control_test.cpp)
    target_link_libraries(kite_control_test kite Eigen3::Eigen)

    add_executable(generic_ocp_test generic_ocp_test.cpp)
    target_link_libraries(generic_ocp_test polymath Eigen3::Eigen)

    #add_executable(cstr_control_test cstr_control_test.cpp)
    #target_link_libraries(cstr_control_test polymath Eigen3::Eigen)

    #add_executable(lqr_test lqr_test.cpp)
    #target_link_libraries(lqr_test polymath Eigen3::Eigen)

endif()
