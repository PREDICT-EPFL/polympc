#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include "gtest/gtest.h"
#include "sqp.hpp"

using namespace sqp;

template <typename _Derived, typename _Scalar, int _VAR_SIZE, int _NUM_EQ=0, int _NUM_INEQ=0, int _NUM_BOX=0>
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

    using ADScalar = Eigen::AutoDiffScalar<grad_t>;
    using ad_var_t = Eigen::Matrix<ADScalar, VAR_SIZE, 1>;
    using ad_eq_t = Eigen::Matrix<ADScalar, NUM_EQ, 1>;
    using ad_ineq_t = Eigen::Matrix<ADScalar, NUM_INEQ, 1>;
    using ad_box_t = Eigen::Matrix<ADScalar, NUM_BOX, 1>;

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

    void constraint_linearized(const var_t& x, A_eq_t& A_eq, b_eq_t& b_eq, A_ineq_t& A_ineq, b_ineq_t& b_ineq, A_box_t& A_box, b_box_t& b_box, b_box_t& l_box, b_box_t& u_box)
    {
        ad_eq_t ad_eq;
        ad_ineq_t ad_ineq;
        ad_box_t ad_box;

        ad_var_t _x = x;
        AD_seed(_x);
        static_cast<_Derived*>(this)->constraint(_x, ad_eq, ad_ineq, ad_box, l_box, u_box);

        for (int i = 0; i < ad_eq.rows(); i++) {
            b_eq[i] = ad_eq[i].value();
            Eigen::Ref<Eigen::MatrixXd> ref = A_eq.row(i);
            ref = ad_eq[i].derivatives().transpose();
        }

        for (int i = 0; i < ad_ineq.rows(); i++) {
            b_ineq[i] = ad_ineq[i].value();
            Eigen::Ref<Eigen::MatrixXd> ref = A_ineq.row(i);
            ref = ad_ineq[i].derivatives().transpose();
        }

        for (int i = 0; i < NUM_BOX; i++) {
            b_box[i] = ad_box[i].value();
            Eigen::Ref<Eigen::MatrixXd> ref = A_box.row(i);
            ref = ad_box[i].derivatives().transpose();
        }
    }
};


struct NLP : public ProblemBase<NLP,
                                double,
                                /* Nx    */2,
                                /* Neq   */1,
                                /* Nineq */1,
                                /* Nbox  */0>  {
    const Scalar a = 1;
    const Scalar b = 100;
    Eigen::Vector2d SOLUTION = {0.7071067812, 0.707106781};

    template <typename DerivedA, typename DerivedB>
    void cost(const DerivedA& x, DerivedB &cst)
    {
        // (a-x)^2 + b*(y-x^2)^2
        cst = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
    }

    template <typename A, typename B, typename C, typename D>
    void constraint(const A& x, B& eq, C& ineq, D& box, b_box_t& l_box, b_box_t& u_box)
    {
        // y >= x
        ineq << x(0) - x(1);
        // x^2 + y^2 == 1
        eq << x.squaredNorm() - 1;
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
                                       /* Nineq */0,
                                       /* Nbox  */0>  {
    const Scalar a = 1;
    const Scalar b = 100;
    Eigen::Vector2d SOLUTION = {1.0, 1.0};

    template <typename DerivedA, typename DerivedB>
    void cost(const DerivedA& x, DerivedB &cst)
    {
        // (a-x)^2 + b*(y-x^2)^2
        cst = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
    }

    template <typename A, typename B, typename C, typename D>
    void constraint(const A& x, B& eq, C& ineq, D& box, b_box_t& l_box, b_box_t& u_box)
    {
        // unconstrained
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


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
