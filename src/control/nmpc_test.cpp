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

    /*
    NumSegments = 2
    POLY_ORDER = 3
    NUM_NODES = 4
    VARX_SIZE = (NumSegments * POLY_ORDER + 1) * VAR_SIZE = 21
    VARU_SIZE = (NumSegments * POLY_ORDER + 1) * NU = 14
    VARP_SIZE = NP = 1
    VAR_SIZE = 3
    NU = 2
    NP = 1
    */

    controller_t robot_controller;
    Eigen::Vector3d x0 = {-1, 0, 0};
    std::cout << "x0 " << x0.transpose() << std::endl;

    var_t sol = robot_controller.solve(x0);

    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ",", "[", "]");
    printf("xy\n");
    for (int i = 0; i < 7; i++) {
       std::cout << sol.segment<2>(3*i).transpose().format(fmt) << ",\n";
    }

    printf("theta\n");
    for (int i = 0; i < 7; i++) {
       std::cout << sol.segment<1>(3*i+2).transpose().format(fmt) << ",\n";
    }

    printf("u\n");
    for (int i = 0; i < 7; i++) {
       std::cout << sol.segment<2>(21+3*i).transpose().format(fmt) << ",\n";
    }

    Eigen::IOFormat fmt2(2);
    std::cout << "dual\n" << robot_controller.solver._lambda.transpose().format(fmt2) << std::endl;

    printf("controller_t size %lu\n", sizeof(controller_t));
    printf("controller_t::cost_colloc_t size %lu\n", sizeof(controller_t::cost_colloc_t));
    printf("controller_t::ode_colloc_t size %lu\n", sizeof(controller_t::ode_colloc_t));
    printf("controller_t::sqp_t size %lu\n", sizeof(controller_t::sqp_t));
    printf("controller_t::sqp_t::qp_t size %lu\n", sizeof(controller_t::sqp_t::qp_t));
    printf("controller_t::sqp_t::qp_solver_t size %lu\n", sizeof(controller_t::sqp_t::qp_solver_t));

    Eigen::Vector3d x0_list[] = {
        {1, 1, 0},
        {-1, -1, 0},
        {1, 0.5, 0},
        {1, 1, -1.5},
        {1, 1, 1.5},
        {10, 10, 0},
        {0, 0, 0},
    };

    for (auto& x0: x0_list) {
        std::cout << "x0 " << x0.transpose() << std::endl;
        var_t sol = robot_controller.solve(x0);
        for (int i = 0; i < 7; i++) {
           std::cout << sol.segment<2>(3*i).transpose().format(fmt) << ",\n";
        }
    }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
