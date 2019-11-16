#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include "gtest/gtest.h"
#define SOLVER_DEBUG
#define SQP_SOLVER_PRINTING
#define QP_SOLVER_PRINTING
#define SOLVER_ASSERT(x) EXPECT_TRUE(x)
#include "solvers/sqp.hpp"

using namespace sqp;

namespace sqp_test_autodiff {

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
    using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

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
            Eigen::Ref<MatX> deriv = ad_eq[i].derivatives().transpose();
            A_eq.row(i) = deriv;
        }

        for (int i = 0; i < ad_ineq.rows(); i++) {
            b_ineq[i] = ad_ineq[i].value();
            Eigen::Ref<MatX> deriv = ad_ineq[i].derivatives().transpose();
            A_ineq.row(i) = deriv;
        }
    }
};


struct SOFCModel : public ProblemBase<SOFCModel,
                                double,
                                /* Nx    */5,
                                /* Neq   */1,
                                /* Nineq */2>  {
    const Scalar delta_air = 1e-5;
    const Scalar delta_cool = 1e-4;
    const Scalar LHV_CH4 = 833.33;      // [J/kg]
    const Scalar N_cell = 70;
    const Scalar F = 96486.00/16e3;    // [C/kg]
    const Scalar Pel_ref = 1000.00;    // [W]
    //const Scalar conv_Lm_to_mols = 7.4356e-4;    // [W]

    //Eigen::Matrix<Scalar, 2, 1> SOLUTION = {0.7071067812, 0.707106781};

    template <typename DerivedA, typename DerivedB>
    void cost(const DerivedA& x, DerivedB &cst)
    {
        // x =[u (4), U_cell(1), theta(15)]
        // cst = -N_cell * x(4) * x(2)/(x(0) * LHV_CH4) + delta_air * pow(x(1), 2) + delta_cool * pow(x(3), 2);
        cst = -1*N_cell * x(4) * x(2) + 1*(1*delta_air * pow(x(1), 2) + 1*delta_cool * pow(x(3), 2))*(x(0) * LHV_CH4);
    }

    template <typename A, typename B, typename C>
    void constraint(const A& x, B& eq, C& ineq, box_t& lbx, box_t& ubx)
    {
        // nu < 0.8, lambda > 4
        ineq << N_cell*x(2) - 0.8*(8*F*x(0)), 4.0*(2.0*x(0)) - x(1);
        //ineq << N_cell*x(2)/(8*F*x(0)) - 0.8, 3.0 - x(1)/(2.0*x(0));
        // x^2 + y^2 == 1
        // eq << x.squaredNorm() - 1;
        // SOFC dynamics, function - f
        eq << Pel_ref - N_cell * x(4) * x(2);//, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        lbx << 1, 85, 0, 0, 0.76;//, 650, -infinity, -infinity, -infinity, 650, -infinity, -infinity, -infinity, -infinity, -infinity, -infinity, -infinity, -infinity, -infinity, -infinity;
        ubx << 7, 200, 50, 40, infinity;//, 750, infinity, infinity, infinity, 790, 890, infinity, infinity, infinity, infinity, infinity, infinity, infinity, infinity, infinity;
    }
};

template <typename Solver>
void callback(void *solver_p)
{
    Solver& s = *static_cast<Solver*>(solver_p);

    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ",", "[", "],");
    std::cout << s.info().iter << " " << s._x.transpose().format(fmt) << s._qp_solver.info().status << " " << s._cost << std::endl;
}

#if 0 // suspended, see issue #13
TEST(SQPTestCase, TestConstrainedRosenbrock) {
    using Solver = SQP<ConstrainedRosenbrock>;
    ConstrainedRosenbrock problem;
    Solver solver;
    Eigen::Vector2d x0, x;
    Eigen::Vector4d y0;
    y0.setZero();

    x0 << 0, 0;
    solver.settings().max_iter = 1000;
    solver.settings().line_search_max_iter = 10;
    // solver.settings().iteration_callback = callback<Solver>;
    solver.solve(problem, x0, y0);

    x = solver.primal_solution();

    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "Solution " << x.transpose() << std::endl;

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}
#endif


TEST(SQPTestCase, TestSOFCModel) {
    using Solver = SQP<SOFCModel>;
    SOFCModel problem;
    Solver solver;
    Eigen::Matrix<double, 5, 1> x;
    Eigen::Matrix<double, 5, 1> x0;
    x0 << 1.75, 150, 18, 30, 0.9;//, 500, 710, 711, 712, 713, 1000, 800, 700, 750, 800, 700, 750, 800, 700, 750;

    solver.settings().max_iter = 500;
    solver.settings().line_search_max_iter = 1;
    //solver._qp_solver.settings().adaptive_rho = true;
    solver._qp_solver.settings().verbose = false;
    solver._qp_solver.settings().max_iter = 1000;
    solver._qp_solver.settings().check_termination = 25;
    solver._qp_solver.settings().adaptive_rho_interval = 25;
    solver.settings().iteration_callback = callback<Solver>;
    solver.solve(problem, x0);

    x = solver.primal_solution();

    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "Solution " << x.transpose() << std::endl;

    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}
/*
struct Rosenbrock : public ProblemBase<Rosenbrock,
                                       double,
                                       2,
                                       0,
                                       0>  {
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
        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        lbx << -infinity, -infinity;
        ubx << infinity, infinity;
    }
};

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
        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        ineq << 1 - x.squaredNorm(),
                  x.squaredNorm() - 2; // 1 <= x0^2 + x1^2 <= 2
        lbx << 0, 0; // x0 > 0 and x1 > 0
        ubx << infinity, infinity;
    }

};

TEST(SQPTestCase, TestSimpleNLP) {
    using Solver = SQP<SimpleNLP>;
    SimpleNLP problem;
    Solver solver;

    // feasible initial point
    Eigen::Vector2d x;
    Eigen::Vector2d x0 = {1.2, 0.1};
    Eigen::Vector4d y0;
    y0.setZero();

    solver.settings().max_iter = 100;
    solver.settings().line_search_max_iter = 4;
    solver.settings().iteration_callback = callback<Solver>;
    solver.solve(problem, x0, y0);

    x = solver.primal_solution();

    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "qp_iter " << solver.info().qp_solver_iter << std::endl;
    std::cout << "Solution " << x.transpose() << std::endl;

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}
*/
} // namespace sqp_test_autodiff
