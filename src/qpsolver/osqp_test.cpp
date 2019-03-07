#include "gtest/gtest.h"
#include "osqp_solver.hpp"

using namespace osqp_solver;

TEST(OSQPTestCase, TestSimpleQP) {
    OSQPSolver::Mn P;
    OSQPSolver::Vn q;
    OSQPSolver::Mmn A;
    OSQPSolver::Vm l, u;
    OSQPSolver::Vn sol, expect;

    P << 4, 1,
         1, 2;
    q << 1, 1;
    A << 1, 1,
         1, 0,
         0, 1;
    l << 1, 0, 0;
    u << 1, 0.7, 0.7;

    OSQPSolver prob;
    prob.setup(P, q, A, l, u);
    prob.solve();
    sol = prob.x;

    expect << 0.3, 0.7;
    ASSERT_TRUE(sol.isApprox(expect, 1e-3f));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
