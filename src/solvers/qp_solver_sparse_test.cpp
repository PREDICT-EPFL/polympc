#include "gtest/gtest.h"
#define QP_SOLVER_PRINTING
#define QP_SOLVER_USE_SPARSE
#include "qp_solver.hpp"

using namespace qp_solver;

class SimpleQP : public QP<2, 3, double>
{
public:
    Eigen::Matrix<double, 2, 1> SOLUTION;
    SimpleQP()
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

TEST(QPSolverTest, testSimpleQP) {
    SimpleQP qp;
    QPSolver<SimpleQP> prob;
    using info_t = qp_solver_info_t<double>;

    prob.settings().max_iter = 1000;
    prob.settings().adaptive_rho = true;

    prob.solve(qp);
    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, info_t::SOLVED);
}

TEST(QPSolverTest, testConjugateGradient) {
    SimpleQP qp;
    QPSolver<SimpleQP, Eigen::ConjugateGradient, Eigen::Lower | Eigen::Upper> prob;
    using info_t = qp_solver_info_t<double>;

    prob.settings().max_iter = 1000;
    prob.settings().adaptive_rho = true;

    prob.solve(qp);
    Eigen::Vector2d sol = prob.primal_solution();

    EXPECT_TRUE(sol.isApprox(qp.SOLUTION, 1e-2));
    EXPECT_LT(prob.iter, prob.settings().max_iter);
    EXPECT_EQ(prob.info().status, info_t::SOLVED);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
