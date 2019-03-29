#include "control/nmpc.hpp"
#include "control/simple_robot_model.hpp"
#include "polynomials/ebyshev.hpp"
#include "qpsolver/osqp_solver.hpp"
#include "qpsolver/sqp.hpp"

#include "gtest/gtest.h"


TEST(NMPCTestCase, TestRobotNMPC)
{
    using Problem = polympc::OCProblem<MobileRobot<double>, Lagrange<double>, Mayer<double>>;
    using Approximation = Chebyshev<3>;

    using controller_t = polympc::nmpc<Problem, Approximation>;
    controller_t robot_controller;

    std::cout << "ode::var_t " << controller_t::ode_colloc_t::var_t::RowsAtCompileTime << std::endl;
    std::cout << "ode::constr_t " << controller_t::ode_colloc_t::constr_t::RowsAtCompileTime << std::endl;
    std::cout << "ode::jacobian_t " << controller_t::ode_colloc_t::jacobian_t::RowsAtCompileTime <<
                 "x" << controller_t::ode_colloc_t::jacobian_t::ColsAtCompileTime << std::endl;
    std::cout << "cost::var_t " << controller_t::cost_colloc_t::hessian_t::RowsAtCompileTime << std::endl;
    std::cout << "cost::hessian_t " << controller_t::cost_colloc_t::hessian_t::RowsAtCompileTime << std::endl;

    sqp::SQP<controller_t> prob;
    controller_t::var_t x0;
    x0.setOnes();
    prob.solve(x0);

    std::cout << "Solution: x = \n" << prob._x.transpose() << std::endl;
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
