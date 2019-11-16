#define QP_SOLVER_PRINTING
#include "gtest/gtest.h"
#include "solvers/qp_solver.hpp"
#include <Eigen/IterativeLinearSolvers>

using namespace qp_solver;

template <typename _Scalar=double>
class _SimpleLP : public QP<1, 1, _Scalar>
{
public:
    Eigen::Matrix<_Scalar, 1, 1> SOLUTION;
    _SimpleLP()
    {
        this->P << 0;
        this->q << 1;
        this->A << 1;
        this->l << -1e6;
        this->u << 1e6;

        this->SOLUTION << -1e6;
    }
};

using TestLP = _SimpleLP<double>;

TEST(QPProblemSets, testSimpleLP) {
    TestLP qp;
    QPSolver<TestLP> prob;

    prob.settings().max_iter = 1000;
    prob.settings().verbose = true;
    prob.settings().alpha = 1.6;
    prob.settings().adaptive_rho = false;
    prob.settings().check_termination = 25;

    prob.setup(qp);
    prob.solve(qp);
    Eigen::VectorXd sol = prob.primal_solution();
    printf("  Solution %.3e \n", sol[0]);

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, SOLVED);
}

template <typename _Scalar=double>
class _SimpleQP : public QP<1, 1, _Scalar>
{
public:
    _SimpleQP()
    {
        this->P << 1;
        this->q << 0;
        this->A << 1;
        this->l << -1;
        this->u << -2;
    }
};

using TestQP = _SimpleQP<double>;

TEST(QPProblemSets, testInfeasibleConstraintsQP) {
    TestQP qp;
    QPSolver<TestQP> prob;

    prob.settings().max_iter = 1000;
    prob.settings().verbose = true;
    prob.settings().alpha = 1.6;
    prob.settings().adaptive_rho = true;
    prob.settings().check_termination = 25;

    prob.setup(qp);
    prob.solve(qp);
    Eigen::VectorXd sol = prob.primal_solution();
    printf("  Solution %.3e \n", sol[0]);

    EXPECT_LT(prob.iter, prob.settings().max_iter);
    Eigen::VectorXd lower = qp.A*sol - qp.l;
    Eigen::VectorXd upper = qp.A*sol - qp.u;
    EXPECT_GE(lower.minCoeff(), -1e-3);
    EXPECT_LE(upper.maxCoeff(), 1e-3);
    EXPECT_EQ(prob.info().status, UNSOLVED);
}

template <typename _Scalar=double>
class _NonconvexQP : public QP<1, 1, _Scalar>
{
public:
    _NonconvexQP()
    {
        this->P << -1;
        this->q << 0;
        this->A << 1;
        this->l << -1;
        this->u << 2;
    }
};

using TestQP1 = _NonconvexQP<double>;

TEST(QPProblemSets, testNonconvexQP) {
    TestQP1 qp;
    QPSolver<TestQP1> prob;

    prob.settings().max_iter = 1000;
    prob.settings().verbose = true;
    prob.settings().alpha = 1.6;
    prob.settings().adaptive_rho = true;
    prob.settings().check_termination = 25;

    prob.setup(qp);
    prob.solve(qp);
    Eigen::VectorXd sol = prob.primal_solution();
    printf("  Solution %.3e \n", sol[0]);

    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, UNSOLVED);
}
