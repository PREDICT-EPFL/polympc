#include "gtest/gtest.h"
#include "sqp.hpp"

using namespace sqp;

template <typename _Scalar, int _VAR_SIZE, int _NUM_EQ=0, int _NUM_INEQ=0, int _NUM_BOX=0>
struct ProblemBase {
    enum {
        VAR_SIZE = _VAR_SIZE,
        NUM_EQ = _NUM_EQ,
        NUM_INEQ = _NUM_INEQ,
        NUM_BOX = _NUM_BOX,
    };

    using Scalar = double;
    using var_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using grad_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using hessian_t = Eigen::Matrix<Scalar, VAR_SIZE, VAR_SIZE>;

    using b_eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using A_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using b_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using A_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;
    using b_box_t = Eigen::Matrix<Scalar, NUM_BOX, 1>;
    using A_box_t = Eigen::Matrix<Scalar, NUM_BOX, VAR_SIZE>;
};

void iteration_callback(const Eigen::MatrixXd &x)
{
    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ",", "[", "],");
    std::cout << x.transpose().format(fmt) << std::endl;
}

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

    EXPECT_TRUE(solver._x.isApprox(SOLUTION, 1e-2));
    EXPECT_LT(solver.iter, solver.settings.max_iter);
}


struct SimpleQP : public ProblemBase<double, 2, 0, 0, 3> {
    Eigen::Matrix2d P;
    Eigen::Vector2d q;
    Eigen::Matrix<Scalar, 3, 2> A;
    Eigen::Vector3d l,u;
    Eigen::Vector2d SOLUTION;

    SimpleQP()
    {
        P << 4, 1,
             1, 2;
        q << 1, 1;
        A << 1, 1,
             1, 0,
             0, 1;
        l << 1, 0, 0;
        u << 1, 0.7, 0.7;
        SOLUTION << 0.3, 0.7;
    }

    void cost(const var_t& x, Scalar &cst)
    {
        cst = 0.5 * x.dot(P*x) + q.dot(x);
    }

    void cost_linearized(const var_t& x, grad_t &grad, Scalar &cst)
    {
        cost(x, cst);
        grad << P*x + q;
    }

    void constraint(const var_t& x, b_eq_t& b_eq, b_ineq_t& b_ineq, b_box_t& b_box, b_box_t& l_box, b_box_t& u_box)
    {
        b_box << A * x;
        l_box << l;
        u_box << u;
    }

    void constraint_linearized(const var_t& x, A_eq_t& A_eq, b_eq_t& b_eq, A_ineq_t& A_ineq, b_ineq_t& b_ineq, A_box_t& A_box, b_box_t& b_box, b_box_t& l_box, b_box_t& u_box)
    {
        constraint(x, b_eq, b_ineq, b_box, l_box, u_box);
        A_box << A;
    }
};

TEST(SQPTestCase, TestSimpleQP) {
    SimpleQP problem;
    SQP<SimpleQP> solver;

    Eigen::Vector2d x0;

    x0 << 0, 0;
    solver.solve(problem, x0);

    EXPECT_TRUE(solver._x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.iter, solver.settings.max_iter);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
