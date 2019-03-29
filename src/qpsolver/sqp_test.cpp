#include "gtest/gtest.h"
#include "sqp.hpp"

using namespace sqp;

struct SimpleNLP_2D {
    enum {
        NX = 2,
        NIEQ = 4,
        NEQ = 0
    };
    using Scalar = double;
    using x_t = Eigen::Vector2d;
    using grad_t = Eigen::Vector2d;
    using constr_t = Eigen::Vector4d;
    using constr_jac_t = Eigen::Matrix<double, 4, 2>;

    // cost function f(x) R^n -> R
    void cost(const x_t& x, Scalar &cst)
    {
        cst = -x(0) -x(1);
    }

    void cost_gradient(const x_t& x, grad_t &grad)
    {
        grad << -1, -1; // solution: [1 1]
        // grad << -1, 0; // solution: [1.41, 0]
        // grad << 0, -1; // solution: [0, 1.41]
        // grad << 1, 1; // solution: [1, 0] or [0, 1]
        // grad << 0, 1; // solution: [0, (1, 1.41)]
    }

    // constraint c(x) R^n -> R^m
    void constraint(const x_t& x, constr_t &c)
    {
        c << -x(0), // -x0 <= 0
             -x(1), // -x1 <= 0
             1 - x.squaredNorm(), // 1 - x0^2 - x1^2 <= 0
             -2 + x.squaredNorm(); // -2 + x0^2 + x1^2 <= 0
    }

    void constraint_jacobian(const x_t& x, constr_jac_t &jac)
    {
        jac << -1, 0,
               0, -1,
               -2*x.transpose(),
               2*x.transpose();
    }
};

TEST(SQPTestCase, TestSimpleNLP) {
    SQP<SimpleNLP_2D> prob;
    prob.settings.max_iter = 100;
    Eigen::Vector2d x0;
    x0 << 1.2, 0.1; // feasible initial point
    prob.solve(x0);

    std::cout << "Solution: x = \n" << prob._x.transpose() << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
