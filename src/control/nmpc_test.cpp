#include "control/nmpc.hpp"
#include "control/simple_robot_model.hpp"
#include "polynomials/ebyshev.hpp"
#include "qpsolver/osqp_solver.hpp"

#include "gtest/gtest.h"


TEST(NMPCTestCase, TestRobotNMPC)
{
    using Problem = polympc::OCProblem<MobileRobot<double>, Lagrange<double>, Mayer<double>>;
    using Approximation = Chebyshev<3>;

    using controller_t = polympc::nmpc<Problem, Approximation>;
    controller_t robot_controller;

    std::cout << controller_t::cost_colloc_t::hessian_t::RowsAtCompileTime << std::endl;
    std::cout << controller_t::ode_colloc_t::jacobian_t::RowsAtCompileTime << std::endl;

    controller_t::qp_t qp;
    controller_t::var_t x = controller_t::var_t::Ones();
    controller_t::Scalar cost;

    robot_controller.construct_subproblem(x, qp);

    std::cout << "P=\n" << qp.P << std::endl;
    std::cout << "q=\n" << qp.q.transpose() << std::endl;
    std::cout << "A=\n" << qp.A << std::endl;
    std::cout << "l=\n" << qp.l.transpose() << std::endl;
    std::cout << "u=\n" << qp.u.transpose() << std::endl;
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
