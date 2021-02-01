cmake_minimum_required(VERSION 3.0)
project(solvers)

enable_testing()

if(BUILD_TESTS)
	find_package(GTest REQUIRED)
	include_directories(${GTEST_INCLUDE_DIRS})
endif()

