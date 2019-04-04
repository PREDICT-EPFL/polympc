#include "control/nmpc.hpp"
#include "control/simple_robot_model.hpp"
#include "polynomials/ebyshev.hpp"
#include "qpsolver/osqp_solver.hpp"

#include "gtest/gtest.h"


TEST(NMPCTestCase, TestRobotNMPC)
{
    using Problem = polympc::OCProblem<MobileRobot<double>, Lagrange<double>, Mayer<double>>;
    using Approximation = Chebyshev<3>; // POLY_ORDER = 3

    using controller_t = polympc::nmpc<Problem, Approximation, int>;
    using var_t = controller_t::var_t;

    controller_t robot_controller;
    Eigen::Vector3d x0 = {10, 10, 3.141};

    var_t sol = robot_controller.solve(x0);
    std::cout << sol.transpose() << std::endl;
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
