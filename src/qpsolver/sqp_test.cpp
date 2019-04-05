#include "gtest/gtest.h"
#include "sqp.hpp"

using namespace sqp;

struct SimpleNLP_2D {
    enum {
        VAR_SIZE = 2,
        NUM_EQ = 0,
        NUM_INEQ = 2,
        NUM_BOX = 1,
    };
    using Scalar = double;
    using x_t = Eigen::Vector2d;
    using grad_t = Eigen::Vector2d;

    using b_eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using A_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using b_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using A_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;
    using b_box_t = Eigen::Matrix<Scalar, NUM_BOX, 1>;
    using A_box_t = Eigen::Matrix<Scalar, NUM_BOX, VAR_SIZE>;

    void cost(const x_t& x, Scalar &cst)
    {
        cst = -x(0) -x(1);
    }

    void cost_linearized(const x_t& x, grad_t &grad, Scalar &cst)
    {
        cost(x, cst);
        grad << -1, -1; // solution: [1 1]
        // grad << -1, 0; // solution: [1.41, 0]
        // grad << 0, -1; // solution: [0, 1.41]
        // grad << 1, 1; // solution: [1, 0] or [0, 1]
        // grad << 0, 1; // solution: [0, (1, 1.41)]
    }

    void constraint(const x_t& x, b_eq_t& b_eq, b_ineq_t& b_ineq, b_box_t& b_box, b_box_t& l_box, b_box_t& u_box)
    {
        b_ineq << -x(0), -x(1); // x0 > 0 and x1 > 0
        b_box << x.squaredNorm(); // 1 <= x0^2 + x1^2 <= 2
        l_box << 1;
        u_box << 2;
    }

    void constraint_linearized(const x_t& x,
                               A_eq_t& A_eq,
                               b_eq_t& b_eq,
                               A_ineq_t& A_ineq,
                               b_ineq_t& b_ineq,
                               A_box_t& A_box,
                               b_box_t& b_box,
                               b_box_t& l_box,
                               b_box_t& u_box)
    {
        constraint(x, b_eq, b_ineq, b_box, l_box, u_box);
        A_ineq << -1,  0,
                   0, -1;
        A_box << 2*x.transpose();
    }
};

TEST(SQPTestCase, TestSimpleNLP) {
    SimpleNLP_2D problem;
    SQP<SimpleNLP_2D> solver;

    Eigen::Vector2d SOLUTION(1, 1);
    Eigen::Vector2d x0;

    x0 << 1.2, 0.1; // feasible initial point
    solver.settings.max_iter = 100;
    solver.solve(problem, x0);

    std::cout << "Feasible x0 " << std::endl;
    std::cout << "iter " << solver.iter << std::endl;
    std::cout << "Solution " << solver._x.transpose() << std::endl;

    EXPECT_TRUE(solver._x.isApprox(SOLUTION, 1e-2));
    EXPECT_LT(solver.iter, solver.settings.max_iter);
}

TEST(SQPTestCase, InfeasibleStart) {
    SimpleNLP_2D problem;
    SQP<SimpleNLP_2D> solver;

    Eigen::Vector2d SOLUTION(1, 1);
    Eigen::Vector2d x0;

    x0 << 2, -1; // infeasible initial point
    solver.settings.max_iter = 100;
    solver.solve(problem, x0);

    std::cout << "Infeasible x0 " << std::endl;
    std::cout << "iter " << solver.iter << std::endl;
    std::cout << "Solution " << solver._x.transpose() << std::endl;

    EXPECT_TRUE(solver._x.isApprox(SOLUTION, 1e-2));
    EXPECT_LT(solver.iter, solver.settings.max_iter);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
