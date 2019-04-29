#include "control/nmpc.hpp"
#include "control/simple_robot_model.hpp"
#include "polynomials/ebyshev.hpp"

#include "gtest/gtest.h"

void iteration_callback(const Eigen::MatrixXd &var)
{
    // std::cout << "SQP iteration callback:" << std::endl;
    // std::cout << var.transpose() << std::endl;
}

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

    printf("controller_t size %lu\n", sizeof(controller_t));
    printf("controller_t::cost_colloc_t size %lu\n", sizeof(controller_t::cost_colloc_t));
    printf("controller_t::ode_colloc_t size %lu\n", sizeof(controller_t::ode_colloc_t));
    printf("controller_t::sqp_t size %lu\n", sizeof(controller_t::sqp_t));
    printf("controller_t::sqp_t::qp_t size %lu\n", sizeof(controller_t::sqp_t::qp_t));
    printf("controller_t::sqp_t::qp_solver_t size %lu\n", sizeof(controller_t::sqp_t::qp_solver_t));

    controller_t robot_controller;
    robot_controller.solver.settings.iteration_callback = iteration_callback;

    Eigen::Vector3d x0_list[] = {
        {-1, 0, 0},
        {-1, -1, 0},
        {-1, -1, 0.7},
        {-1, -1, 1.6},
        {-10, -10, 0},
        {1, 1, 0},
        {10, 10, 0},
        {0, 1, 0.7},
        {0, 1, 0},
        {0, 0, 0},
    };

    for (auto& x0: x0_list) {
        std::cout << "x0 " << x0.transpose() << std::endl;

        // bounds
        controller_t::State xu, xl;
        xu << 10, 10, 1e20;
        xl << -xu;
        controller_t::Control uu, ul;
        uu << 10, 1;
        ul << 0, -1;

        var_t sol = robot_controller.solve(x0, xl, xu, ul, uu);

        Eigen::IOFormat fmt(4, 0, ", ", ",", "[", "]");
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
           std::cout << sol.segment<2>(21+2*i).transpose().format(fmt) << ",\n";
        }
        printf("p\n");
        std::cout << sol.tail<1>().format(fmt) << ",\n";

        printf("iter %d\n", robot_controller.solver.iter);
        printf("dual\n");
        std::cout << "  ode   " << robot_controller.solver._lambda.template segment<controller_t::VARX_SIZE>(0).transpose().format(fmt) << std::endl;
        std::cout << "  x0    " << robot_controller.solver._lambda.template segment<controller_t::NX>(controller_t::VARX_SIZE).transpose().format(fmt) << std::endl;
        std::cout << "  x     " << robot_controller.solver._lambda.template segment<controller_t::VARX_SIZE-controller_t::NX>(controller_t::VARX_SIZE+controller_t::NX).transpose().format(fmt) << std::endl;
        std::cout << "  u     " << robot_controller.solver._lambda.template segment<controller_t::VARU_SIZE>(2*controller_t::VARX_SIZE).transpose().format(fmt) << std::endl;
    }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
