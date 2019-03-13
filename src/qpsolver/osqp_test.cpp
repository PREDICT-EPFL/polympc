#include "gtest/gtest.h"
#include "osqp_solver.hpp"

using namespace osqp_solver;

TEST(OSQPTestCase, TestSimpleQP) {
    using SimpleQP = QP<2, 3, double>;
    OSQPSolver<SimpleQP> prob;
    Eigen::Vector2d sol, expect;

    SimpleQP qp;
    qp.P << 4, 1,
            1, 2;
    qp.q << 1, 1;
    qp.A << 1, 1,
            1, 0,
            0, 1;
    qp.l << 1, 0, 0;
    qp.u << 1, 0.7, 0.7;

    OSQPSolver<SimpleQP>::Settings settings;
    settings.rho = 0.1;
    settings.max_iter = 50;
    settings.eps_rel = 1e-4f; // set below isApprox() threshold
    settings.eps_abs = 1e-4f;

    prob.solve(qp, settings);
    sol = prob.x;

    // solution
    expect << 0.3, 0.7;
    EXPECT_TRUE(sol.isApprox(expect, 1e-3));

    // check feasibility (with some epsilon margin)
    Eigen::Vector3d lower = qp.A*sol - qp.l;
    Eigen::Vector3d upper = qp.A*sol - qp.u;
    EXPECT_GE(lower.minCoeff(), -1e-3);
    EXPECT_LE(upper.maxCoeff(), 1e-3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
