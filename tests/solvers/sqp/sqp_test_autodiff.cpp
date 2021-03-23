#include "utils/helpers.hpp"
#include "solvers/nlproblem.hpp"
#include "solvers/sqp_base.hpp"
#include "gtest/gtest.h"

/** create solver */
template<typename Problem> class SQPSolver;
template<typename Problem>
class SQPSolver : public SQPBase<SQPSolver<Problem>, Problem>
{
public:
    using Base = SQPBase<SQPSolver<Problem>, Problem>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;
    using typename Base::nlp_hessian_t;

    Eigen::EigenSolver<nlp_hessian_t> es;

    /** implement regularisation for the Hessian: eigenvalue mirroring */
    EIGEN_STRONG_INLINE void hessian_regularisation_dense_impl(Eigen::Ref<nlp_hessian_t> lag_hessian) noexcept
    {
        Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>  Deig;
        es.compute(lag_hessian);

        scalar_t minEigValue = es.eigenvalues().real().minCoeff();
        if ( minEigValue <= 0)
        {
            Deig = es.eigenvalues().real().asDiagonal();
            for (int i = 0; i < Deig.rows(); i++)
            {
                if (Deig(i, i) <= 0) { Deig(i, i) = -1 * Deig(i, i) + 0.1; } //Mirror regularization
            }

            lag_hessian.noalias() = (es.eigenvectors().real()) * Deig* (es.eigenvectors().real().transpose()); //V*D*V^-1 with V^-1 ~= V'
        }
    }

};

// Constrained Rosenbrock Function
POLYMPC_FORWARD_NLP_DECLARATION(/*Name*/ ConstrainedRosenbrock, /*NX*/ 2, /*NE*/1, /*NI*/0, /*NP*/0, /*Type*/double);
class ConstrainedRosenbrock : public ProblemBase<ConstrainedRosenbrock>
{
public:
    const scalar_t a = 1;
    const scalar_t b = 100;
    Eigen::Matrix<scalar_t, 2, 1> SOLUTION = {0.7864, 0.6177};

    template<typename T>
    EIGEN_STRONG_INLINE void cost_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
        //Eigen::Array<T,2,1> lox;
        // (a-x)^2 + b*(y-x^2)^2
        //cost = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
        polympc::ignore_unused_var(p);
        cost = (a - x(0)) * (a - x(0)) + b * (x(1) - x(0)*x(0)) * (x(1) - x(0)*x(0));
    }

    template<typename T>
    EIGEN_STRONG_INLINE void equality_constraints_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                       Eigen::Ref<constraint_t<T>> constraint) const noexcept
    {
        // x^2 + y^2 == 1
        constraint << x.squaredNorm() - 1;
        polympc::ignore_unused_var(p);
    }
};

TEST(SQPTestCase, TestConstrainedRosenbrock)
{
    // will be using the default
    using Solver = SQPSolver<ConstrainedRosenbrock>;
    ConstrainedRosenbrock problem;
    Solver solver;
    Solver::nlp_variable_t x0, x;
    Solver::nlp_dual_t y0;
    y0.setZero();

    x0 << 2.01, 1.01;
    solver.settings().max_iter = 50;
    solver.settings().line_search_max_iter = 5;
    solver.solve(x0, y0);

    x = solver.primal_solution();

    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "Solution " << x.transpose() << std::endl;

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

// Unconstrained Rosenbrock Function
POLYMPC_FORWARD_NLP_DECLARATION(/*Name*/ Rosenbrock, /*NX*/ 2, /*NE*/0, /*NI*/0, /*NP*/0, /*Type*/double);
class Rosenbrock : public ProblemBase<Rosenbrock>
{
public:
    const scalar_t a = 1;
    const scalar_t b = 100;
    Eigen::Matrix<scalar_t, 2, 1> SOLUTION = {1.0, 1.0};

    template<typename T>
    EIGEN_STRONG_INLINE void cost_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
        //Eigen::Array<T,2,1> lox;
        // (a-x)^2 + b*(y-x^2)^2
        //cost = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
        polympc::ignore_unused_var(p);
        cost = (a - x(0)) * (a - x(0)) + b * (x(1) - x(0)*x(0)) * (x(1) - x(0)*x(0));
    }
};

TEST(SQPTestCase, TestRosenbrock) {
    // will be using the default
    using Solver = SQPSolver<Rosenbrock>;
    Rosenbrock problem;
    Solver solver;
    Solver::nlp_variable_t x0, x;
    Solver::nlp_dual_t y0;
    y0.setZero();

    x0 << 2.01, 1.01;
    solver.settings().max_iter = 50;
    solver.settings().line_search_max_iter = 5;
    solver.solve(x0, y0);

    x = solver.primal_solution();

    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "Solution " << x.transpose() << std::endl;

    EXPECT_TRUE(x.isApprox(problem.SOLUTION, 1e-2));
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}

// here wait for refactoring of inequality constraints: coming soon
#if 0
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

#endif
