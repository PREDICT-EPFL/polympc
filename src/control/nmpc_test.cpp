#include "control/nmpc.hpp"
#include "control/simple_robot_model.hpp"
#include "polynomials/ebyshev.hpp"
#include "qpsolver/osqp_solver.hpp"

#include "gtest/gtest.h"


TEST(NMPCTestCase, TestRobotNMPC)
{
    using Problem = polympc::OCProblem<MobileRobot<double>, Lagrange<double>, Mayer<double>>;
    using Approximation = Chebyshev<3>;

    using SimpleQP = osqp_solver::QP<2, 3, double>;
    using Solver = osqp_solver::OSQPSolver<SimpleQP>;

    polympc::nmpc<Problem, Approximation, Solver> robot_controller;
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
