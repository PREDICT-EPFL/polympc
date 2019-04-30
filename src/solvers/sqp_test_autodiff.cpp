#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include "gtest/gtest.h"
#include "sqp.hpp"

using namespace sqp;

template <typename _Derived, typename _Scalar, int _VAR_SIZE, int _NUM_EQ=0, int _NUM_INEQ=0>
struct ProblemBase {
    enum {
        VAR_SIZE = _VAR_SIZE,
        NUM_EQ = _NUM_EQ,
        NUM_INEQ = _NUM_INEQ,
    };

    using Scalar = double;
    using var_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using grad_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using hessian_t = Eigen::Matrix<Scalar, VAR_SIZE, VAR_SIZE>;

    using b_eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using A_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using b_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using A_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;
    using box_t = var_t;

    using ADScalar = Eigen::AutoDiffScalar<grad_t>;
    using ad_var_t = Eigen::Matrix<ADScalar, VAR_SIZE, 1>;
    using ad_eq_t = Eigen::Matrix<ADScalar, NUM_EQ, 1>;
    using ad_ineq_t = Eigen::Matrix<ADScalar, NUM_INEQ, 1>;

    template <typename vec>
    void AD_seed(vec &x)
    {
        for (int i=0; i<x.rows(); i++) {
            x[i].derivatives().coeffRef(i) = 1;
        }
    }

    void cost_linearized(const var_t& x, grad_t &grad, Scalar &cst)
    {
        ad_var_t _x = x;
        ADScalar _cst;
        AD_seed(_x);
        /* Static polymorphism using CRTP */
        static_cast<_Derived*>(this)->cost(_x, _cst);
        cst = _cst.value();
        grad = _cst.derivatives();
    }

    void constraint_linearized(const var_t& x, A_eq_t& A_eq, b_eq_t& b_eq, A_ineq_t& A_ineq, b_ineq_t& b_ineq, box_t& lbx, box_t& ubx)
    {
        ad_eq_t ad_eq;
        ad_ineq_t ad_ineq;

        ad_var_t _x = x;
        AD_seed(_x);
        static_cast<_Derived*>(this)->constraint(_x, ad_eq, ad_ineq, lbx, ubx);

        for (int i = 0; i < ad_eq.rows(); i++) {
            b_eq[i] = ad_eq[i].value();
            Eigen::Ref<Eigen::MatrixXd> deriv = ad_eq[i].derivatives().transpose();
            A_eq.row(i) = deriv;
        }

        for (int i = 0; i < ad_ineq.rows(); i++) {
            b_ineq[i] = ad_ineq[i].value();
            Eigen::Ref<Eigen::MatrixXd> deriv = ad_ineq[i].derivatives().transpose();
            A_ineq.row(i) = deriv;
        }
    }
};


struct NLP : public ProblemBase<NLP,
                                double,
                                /* Nx    */2,
                                /* Neq   */1,
                                /* Nineq */1>  {
    const Scalar a = 1;
    const Scalar b = 100;
    Eigen::Vector2d SOLUTION = {0.7071067812, 0.707106781};

    template <typename DerivedA, typename DerivedB>
    void cost(const DerivedA& x, DerivedB &cst)
    {
        // (a-x)^2 + b*(y-x^2)^2
        cst = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
    }

    template <typename A, typename B, typename C>
    void constraint(const A& x, B& eq, C& ineq, box_t& lbx, box_t& ubx)
    {
        // y >= x
        ineq << x(0) - x(1);
        // x^2 + y^2 == 1
        eq << x.squaredNorm() - 1;

        lbx << -INFINITY, -INFINITY;
        ubx << INFINITY, INFINITY;
    }
};


void iteration_callback(const Eigen::MatrixXd &x)
{
    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ",", "[", "],");
    std::cout << x.transpose().format(fmt) << std::endl;
}

TEST(SQPTestCase, TestNLP) {
    NLP problem;
    SQP<NLP> solver;
    Eigen::Vector2d x0;

    x0 << 0, 0;
    solver.settings.max_iter = 1000;
    solver.settings.line_search_max_iter = 10;
    // solver.settings.iteration_callback = iteration_callback;
    solver.solve(problem, x0);

    std::cout << "iter " << solver.iter << std::endl;
    std::cout << "Solution " << solver._x.transpose() << std::endl;

    EXPECT_TRUE(solver._x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.iter, solver.settings.max_iter);
}


struct Rosenbrock : public ProblemBase<Rosenbrock,
                                       double,
                                       /* Nx    */2,
                                       /* Neq   */0,
                                       /* Nineq */0>  {
    const Scalar a = 1;
    const Scalar b = 100;
    Eigen::Vector2d SOLUTION = {1.0, 1.0};

    template <typename DerivedA, typename DerivedB>
    void cost(const DerivedA& x, DerivedB &cst)
    {
        // (a-x)^2 + b*(y-x^2)^2
        cst = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
    }

    template <typename A, typename B, typename C>
    void constraint(const A& x, B& eq, C& ineq, box_t& lbx, box_t& ubx)
    {
        // unconstrained
        lbx << -INFINITY, -INFINITY;
        ubx << INFINITY, INFINITY;
    }
};

TEST(SQPTestCase, TestRosenbrock) {
    Rosenbrock problem;
    SQP<Rosenbrock> solver;
    Eigen::Vector2d x0;

    x0 << 0, 0;
    solver.settings.max_iter = 1000;
    // solver.settings.line_search_max_iter = 4;
    // solver.settings.iteration_callback = iteration_callback;
    solver.solve(problem, x0);

    std::cout << "iter " << solver.iter << std::endl;
    std::cout << "Solution " << solver._x.transpose() << std::endl;

    EXPECT_TRUE(solver._x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.iter, solver.settings.max_iter);
}

struct SimpleNLP : ProblemBase<SimpleNLP, double, 2, 0, 2> {
    var_t SOLUTION = {1, 1};

    template <typename A, typename B>
    void cost(const A& x, B& cst)
    {
        cst = -x(0) -x(1);
    }

    template <typename A, typename B, typename C>
    void constraint(const A& x, B& eq, C& ineq, var_t& lbx, var_t& ubx)
    {
        ineq << 1 - x.squaredNorm(),
                  x.squaredNorm() - 2; // 1 <= x0^2 + x1^2 <= 2
        lbx << 0, 0; // x0 > 0 and x1 > 0
        ubx << INFINITY, INFINITY;
    }

};

TEST(SQPTestCase, TestSimpleNLP) {
    SimpleNLP problem;
    SQP<SimpleNLP> solver;

    // feasible initial point
    Eigen::Vector2d x0 = {1.2, 0.1};

    solver.settings.max_iter = 100;
    solver.settings.line_search_max_iter = 4;
    solver.settings.iteration_callback = iteration_callback;
    solver.solve(problem, x0);

    std::cout << "iter " << solver.iter << std::endl;
    std::cout << "qp_iter " << solver._qp_iter << std::endl;
    std::cout << "Solution " << solver._x.transpose() << std::endl;

    EXPECT_TRUE(solver._x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.iter, solver.settings.max_iter);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
