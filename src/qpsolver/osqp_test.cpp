#include "gtest/gtest.h"
#include "osqp_solver.hpp"
#include <Eigen/IterativeLinearSolvers>

using namespace osqp_solver;

template <typename _Scalar=double>
class _SimpleQP : public QP<2, 3, _Scalar>
{
public:
    Eigen::Matrix<_Scalar, 2, 1> SOLUTION;
    _SimpleQP()
    {
        this->P << 4, 1,
                   1, 2;
        this->q << 1, 1;
        this->A << 1, 1,
                   1, 0,
                   0, 1;
        this->l << 1, 0, 0;
        this->u << 1, 0.7, 0.7;

        this->SOLUTION << 0.3, 0.7;
    }
};

using SimpleQP = _SimpleQP<double>;

TEST(QPSolverTest, testSimpleQP) {
    SimpleQP qp;
    OSQPSolver<SimpleQP> prob;

    prob.settings.max_iter = 1000;

    prob.solve(qp);
    Eigen::Vector2d sol = prob.x;

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(prob.iter, prob.settings.max_iter);
}


TEST(QPSolverTest, testSinglePrecisionFloat) {
    using SimpleQPf = _SimpleQP<float>;
    SimpleQPf qp;
    OSQPSolver<SimpleQPf> prob;

    prob.solve(qp);
    Eigen::Vector2f sol = prob.x;

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(prob.iter, prob.settings.max_iter);
}

TEST(QPSolverTest, testConstraintViolation) {
    SimpleQP qp;
    OSQPSolver<SimpleQP> prob;

    prob.settings.eps_rel = 1e-4f;
    prob.settings.eps_abs = 1e-4f;

    prob.solve(qp);
    Eigen::Vector2d sol = prob.x;

    // check feasibility (with some epsilon margin)
    Eigen::Vector3d lower = qp.A*sol - qp.l;
    Eigen::Vector3d upper = qp.A*sol - qp.u;
    EXPECT_GE(lower.minCoeff(), -1e-3);
    EXPECT_LE(upper.maxCoeff(), 1e-3);
}

TEST(QPSolverTest, testAdaptiveRho) {
    SimpleQP qp;
    OSQPSolver<SimpleQP> prob;

    prob.settings.adaptive_rho = false;
    prob.settings.adaptive_rho_interval = 10;

    prob.solve(qp);
}

TEST(QPSolverTest, testAdaptiveRhoImprovesConvergence) {
    SimpleQP qp;
    OSQPSolver<SimpleQP> prob;

    prob.settings.warm_start = false;
    prob.settings.max_iter = 1000;
    prob.settings.rho = 0.1;

    // solve whithout adaptive rho
    prob.settings.adaptive_rho = false;
    prob.solve(qp);
    int prev_iter = prob.iter;

    // solve with adaptive rho
    prob.settings.adaptive_rho = true;
    prob.settings.adaptive_rho_interval = 10;
    prob.solve(qp);

    EXPECT_LT(prob.iter, prob.settings.max_iter);
    EXPECT_LT(prob.iter, prev_iter); // adaptive rho should improve :)
}

TEST(QPSolverTest, testConjugateGradientLinearSolver)
{
    SimpleQP qp;
    OSQPSolver<SimpleQP, Eigen::ConjugateGradient, Eigen::Lower | Eigen::Upper> prob;

    prob.solve(qp);
    Eigen::Vector2d sol = prob.x;

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(prob.iter, prob.settings.max_iter); // convergence test
}

TEST(QPSolverTest, TestConstraint) {
    using qp_t = QP<5, 5, double>;
    using solver_t = OSQPSolver<qp_t>;
    solver_t prob;

    qp_t qp;
    qp.P.setIdentity();
    qp.q.setConstant(-1);
    qp.A.setIdentity();

    int type_expect[5];
    qp.l(0) = -1e+17;
    qp.u(0) = 1e+17;
    type_expect[0] = solver_t::LOOSE_BOUNDS;
    qp.l(1) = -101;
    qp.u(1) = 1e+17;
    type_expect[1] = solver_t::INEQUALITY_CONSTRAINT;
    qp.l(2) = -1e+17;
    qp.u(2) = 123;
    type_expect[2] = solver_t::INEQUALITY_CONSTRAINT;
    qp.l(3) = -1;
    qp.u(3) = 1;
    type_expect[3] = solver_t::INEQUALITY_CONSTRAINT;
    qp.l(4) = 42;
    qp.u(4) = 42;
    type_expect[4] = solver_t::EQUALITY_CONSTRAINT;

    prob.solve(qp);

    for (int i = 0; i < qp.l.rows(); i++) {
        EXPECT_EQ(prob.constr_type[i], type_expect[i]);
    }
}

TEST(QPSolverTest, QP2) {
    using qp_t = QP<2, 4, double>;
    OSQPSolver<qp_t> prob;
    Eigen::Vector2d sol, expect;

    qp_t qp;
    qp.P << 1, 0,
            0, 1;
    qp.q << -1, -1;
    qp.A << -1, 0,
            0, -1,
            2.4, 0.2,
            -2.4, -0.2;
    qp.l << -1e+16, -1e+16, -1e+16, -1e+16;
    qp.u << 1.2,  0.1, 0.45, 0.55;

    prob.settings.max_iter = 1000;

    prob.solve(qp);
    sol = prob.x;

    // solution
    expect << 0.10972805, 0.92581067;
    EXPECT_TRUE(sol.isApprox(expect, 1e-3));
    EXPECT_LT(prob.iter, prob.settings.max_iter);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
