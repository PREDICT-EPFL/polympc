#include "control/problem.hpp"
#include "control/simple_robot_model.hpp"
#include "gtest/gtest.h"


TEST(OCPTestCase, TestSimpleOCP)
{
    polympc::OCProblem<MobileRobot<double>, Lagrange<double>, Mayer<double>> problem;
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}




