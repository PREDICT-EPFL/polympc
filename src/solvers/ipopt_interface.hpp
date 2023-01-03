// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef IPOPT_INTERFACE_HPP
#define IPOPT_INTERFACE_HPP

#include <memory>
#include <iostream>

#include "Eigen/Core"
#include "utils/helpers.hpp"

#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "IpTNLP.hpp"

/** Ipopt info */
struct ipopt_info_t {
    int iter;
    Ipopt::ApplicationReturnStatus status;
};


template<typename Problem>
class IpoptAdapter : public Ipopt::TNLP
{
public:
    IpoptAdapter()
    {
        /** set default constraints */
        const scalar_t INF = std::numeric_limits<scalar_t>::infinity();

        m_lbx = nlp_variable_t::Constant(static_cast<scalar_t>(-INF));
        m_ubx = nlp_variable_t::Constant(static_cast<scalar_t>( INF));
        m_lbg = nlp_ineq_constraints_t::Constant(static_cast<scalar_t>(-INF));
        m_ubg = nlp_ineq_constraints_t::Constant(static_cast<scalar_t>( INF));

        m_x.setZero();
        m_lam.setZero();

        if(Problem::is_sparse)
        {
            m_H.resize(Problem::VAR_SIZE, Problem::VAR_SIZE);
            m_A.resize(Problem::NUM_EQ + Problem::NUM_INEQ, Problem::VAR_SIZE);

            /** run sensitivity computation once to obtain sparsity patterns */
            problem.lagrangian_gradient_hessian(m_x, m_p, m_lam, m_cost, m_h, m_H, m_dlag, m_g, m_A);
        }
    }
    virtual ~IpoptAdapter() = default;

    static constexpr int num_variables = Problem::VAR_SIZE;
    static constexpr int num_constraints = Problem::NUM_EQ + Problem::NUM_INEQ;


    /** get main function from the derived class */
    using nlp_variable_t    = typename Problem::nlp_variable_t;
    using nlp_constraints_t = typename Problem::nlp_constraints_t;
    using nlp_eq_constraints_t   = typename Problem::nlp_eq_constraints_t;
    using nlp_ineq_constraints_t = typename Problem::nlp_ineq_constraints_t;
    using nlp_eq_jacobian_t = typename Problem::nlp_eq_jacobian_t;
    using nlp_jacobian_t    = typename Problem::nlp_jacobian_t;
    using nlp_hessian_t     = typename Problem::nlp_hessian_t;
    using nlp_cost_t        = typename Problem::scalar_t;
    using nlp_dual_t        = typename Problem::nlp_dual_t;
    using scalar_t          = typename Problem::scalar_t;
    using parameter_t       = typename Problem::static_parameter_t;

    /** instantiate the problem */
    Problem problem;

    nlp_hessian_t     m_H;            // Hessian of Lagrangian
    nlp_variable_t    m_h, m_dlag;            // Gradient of the cost function / lagrangian
    nlp_variable_t    m_x;            // variable primal
    nlp_dual_t        m_lam; // dual variable
    nlp_jacobian_t    m_A;            // equality constraints Jacobian
    nlp_constraints_t m_g;     // equality/inequality constraints evaluated
    parameter_t       m_p = parameter_t::Zero(); // problem parameters
    scalar_t          m_cost{0};
    nlp_variable_t    m_lbx, m_ubx;
    nlp_ineq_constraints_t m_lbg, m_ubg;

    /** workaround for dense and sparse matrices: get pointers to the data and sparsity patterns */
    // data
    template<typename MatrixType, int T = Problem::MATRIXFMT>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    get_data(MatrixType &mat, scalar_t* data, const bool symmetric = false) const noexcept
    {
        if(symmetric)
        {
            int ind = 0;
            for (int k = 0; k < mat.outerSize(); ++k)
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(mat, k); it; ++it)
                {
                    if(it.row() >= it.col())
                    {
                        data[ind] = it.value();
                        ++ind;
                    }
                }
        }
        else
            data = mat.derived().valuePtr();
    }

    template<typename MatrixType, int T = Problem::MATRIXFMT>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
    get_data(Eigen::MatrixBase<MatrixType> &mat, scalar_t* data, const bool symmetric = false) const noexcept
    {
        if(symmetric)
        {
            int ind = 0;
            for (int j = 0; j < mat.cols(); ++j)
                for (int i = 0; i < mat.rows(); ++i)
                {
                    if(i >= j)
                    {
                        data[ind] = mat(i,j);
                        ++ind;
                    }
                }
        }
        else
        {
            int ind = 0;
            for (int j = 0; j < mat.cols(); ++j)
                for (int i = 0; i < mat.rows(); ++i)
                {
                    data[ind] = mat(i,j);
                    ++ind;
                }
        }
    }

    // sparsity
    template<typename MatrixType, int T = Problem::MATRIXFMT>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    get_sparsity(const MatrixType &mat, Ipopt::Index* irow, Ipopt::Index* jcol, bool symmetric = false) const noexcept
    {
        eigen_assert(irow != nullptr);
        eigen_assert(jcol != nullptr);

        int ind = 0;
        if(symmetric)
        {
            for (int k = 0; k < mat.outerSize(); ++k)
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(mat, k); it; ++it)
                {
                    if(it.row() >= it.col())
                    {
                        irow[ind] = it.row();
                        jcol[ind] = it.col();
                        ++ind;
                    }
                }
        }
        else
        {
            for (int k = 0; k < mat.outerSize(); ++k)
                for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(mat, k); it; ++it, ind++)
                {
                    irow[ind] = it.row();
                    jcol[ind] = it.col();
                }
        }
    }

    template<typename MatrixType, int T = Problem::MATRIXFMT>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
    get_sparsity(const Eigen::Ref<const MatrixType> &mat, Ipopt::Index* irow, Ipopt::Index* jcol, const bool symmetric = false) const noexcept
    {
        //eigen_assert(sizeof(irow) / sizeof(Ipopt::Index) == mat.size());
        //eigen_assert(sizeof(jcol) / sizeof(Ipopt::Index) == mat.size());

        int ind = 0;
        if(symmetric)
        {
            for (int j = 0; j < mat.cols(); ++j)
            {
                for (int i = 0; i < mat.rows(); ++i)
                {
                    if( i >= j )
                    {
                        irow[ind] = i;
                        jcol[ind] = j;
                        ++ind;
                    }
                }
            }
        }
        else
        {

            for (int j = 0; j < mat.cols(); ++j)
                for (int i = 0; i < mat.rows(); ++i, ++ind)
                {
                    irow[ind] = i;
                    jcol[ind] = j;
                }
        }

    }

    virtual bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                              Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style) override
    {
        n = num_variables;
        m = num_constraints;
        nnz_jac_g = problem.nnz_jacobian();
        nnz_h_lag = problem.nnz_lag_hessian();
        index_style = Ipopt::TNLP::C_STYLE;

        /**
        std::cout << "nlp_info: " << "\n";
        std::cout << "nnz_jac: "  << nnz_jac_g << "\n"
                  << "nnz_hes: "  << nnz_h_lag << "\n";
                  */

        return true;
    }

    virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                                 Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u) override
    {
        // n and m are whatever we set in get_nlp_info
        eigen_assert(n == num_variables);
        eigen_assert(m == num_constraints);

        // x
        Eigen::Map<nlp_variable_t>(x_l, num_variables, 1) = m_lbx;
        Eigen::Map<nlp_variable_t>(x_u, num_variables, 1) = m_ubx;

        // g
        if(Problem::NUM_EQ > 0)
        {
            Eigen::Map<nlp_eq_constraints_t>(g_l, Problem::NUM_EQ, 1) = nlp_eq_constraints_t::Zero(Problem::NUM_EQ, 1);
            Eigen::Map<nlp_eq_constraints_t>(g_u, Problem::NUM_EQ, 1) = nlp_eq_constraints_t::Zero(Problem::NUM_EQ, 1);
        }

        if(Problem::NUM_INEQ > 0)
        {
            Eigen::Map<nlp_ineq_constraints_t>(g_l + Problem::NUM_EQ, Problem::NUM_INEQ, 1) = m_lbg;
            Eigen::Map<nlp_ineq_constraints_t>(g_u + Problem::NUM_EQ, Problem::NUM_INEQ, 1) = m_ubg;
        }

        return true;
    }

    virtual bool get_starting_point( Ipopt::Index n, bool init_x, Ipopt::Number* x, bool init_z,
                             Ipopt::Number* z_L, Ipopt::Number* z_U, Ipopt::Index m, bool init_lambda, Ipopt::Number* lambda) override
    {
        if(init_x)
        {
            Eigen::Map<nlp_variable_t>(x, num_variables, 1) = m_x;
        }

        if(init_z)
        {
            Eigen::Map<nlp_variable_t>(z_L, num_variables, 1) = m_lam.template tail<num_variables>().cwiseMin(0.0).cwiseAbs().eval();
            Eigen::Map<nlp_variable_t>(z_U, num_variables, 1) = m_lam.template tail<num_variables>().cwiseMax(0.0);
        }

        if(init_lambda)
        {
            Eigen::Map<nlp_constraints_t>(lambda, num_constraints, 1) = m_lam.template head<num_constraints>();
        }

        return true;
    }

    // cost
    virtual bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& obj_value) override
    {
        eigen_assert(n == num_variables);
        problem.cost(Eigen::Map<const nlp_variable_t>(x), m_p, obj_value);
        return true;
    }

    // cost gradient
    virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* grad_f) override
    {
        eigen_assert(n == num_variables);
        scalar_t cost{0};
        problem.cost_gradient(Eigen::Map<const nlp_variable_t>(x), m_p, cost, Eigen::Map<nlp_variable_t>(grad_f));
        return true;
    }

    //constraints
    virtual bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Number* g) override
    {
        //std::cout << "eval_g \n";
        eigen_assert(n == num_variables);
        eigen_assert(m == num_constraints);

        Eigen::Map<const nlp_variable_t > var(x);
        Eigen::Map<nlp_constraints_t> constraints(g);

        problem.equalities(var, m_p, constraints.template head<Problem::NUM_EQ>());
        problem.inequalities(var, m_p, constraints.template tail<Problem::NUM_INEQ>());

        return true;
    }

    // constraints jacobian
    virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m,
                    Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index* jCol, Ipopt::Number* values) override
    {
        eigen_assert(n == num_variables);
        eigen_assert(m == num_constraints);

        if( values == NULL )
        {
            // Copy the pre-computed jacobian structure into the Ipopt variables
            get_sparsity<nlp_jacobian_t>(m_A, iRow, jCol);

            /**
            std::cout << "Jacobian sparsity: \n";
            for(int i = 0; i < nele_jac; ++i)
            {
                std::cout << "i : " << i << " [ " << iRow[i] << " , " << jCol[i] << " ] \n";
            }
            */
        }
        else
        {
            problem.constraints_linearised(Eigen::Map<const nlp_variable_t>(x), m_p, m_g, m_A);
            get_data(m_A, values);
        }

        return true;
    }

    virtual bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number obj_factor, Ipopt::Index m,
                const Ipopt::Number* lambda, bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow, Ipopt::Index* jCol, Ipopt::Number* values) override
    {
        eigen_assert(n == num_variables);
        eigen_assert(m == num_constraints);

        if( values == NULL )
        {
            // Copy the pre-computed hessian structure into the Ipopt variables
            get_sparsity<nlp_hessian_t>(m_H, iRow, jCol, true); // uses the symmetric version

            /**
            std::cout << "Hessian sparsity: \n";
            for(int i = 0; i < nele_hess; ++i)
            {
                std::cout << "i : " << i << " [ " << iRow[i] << " , " << jCol[i] << " ] \n";
            }
            */
        }
        else
        {
            // // return the values. This is a symmetric matrix, fill the lower left
            // // triangle only
            problem.lagrangian_gradient_hessian(Eigen::Map<const nlp_variable_t>(x), m_p,
                                                Eigen::Map<const nlp_dual_t>(lambda), m_cost,
                                                m_h, m_H, m_dlag, m_g, m_A, obj_factor);

            get_data(m_H, values, true);
        }

        return true;
    }

    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number* x, const Ipopt::Number* z_L, const Ipopt::Number* z_U,
                           Ipopt::Index m,const Ipopt::Number* g, const Ipopt::Number* lambda, Ipopt::Number obj_value, const Ipopt::IpoptData* ip_data,
                           Ipopt::IpoptCalculatedQuantities* ip_cq) override
    {
        m_x   = Eigen::Map<const nlp_variable_t>(x);
        m_lam = Eigen::Map<const nlp_dual_t>(lambda);
        m_cost = obj_value;

        /** @todo: Fill in the rest of the solution struct */
    }
};


template<typename Problem>
class IpoptInterface
{
public:
    using IpProblem = IpoptAdapter<Problem>;

private:
    Ipopt::SmartPtr<Ipopt::IpoptApplication> m_app;
    Ipopt::SmartPtr<IpProblem> m_problem;
    ipopt_info_t m_info;

public:
    IpoptInterface()
    {
        m_app = IpoptApplicationFactory();
        m_problem = new IpProblem();

        m_app->Options()->SetNumericValue("tol", 1e-6);
        m_app->Options()->SetIntegerValue("print_level", 0);
        m_app->Options()->SetStringValue("mu_strategy", "adaptive");
        m_app->Options()->SetIntegerValue("max_iter", 100);

        // initialise
        Ipopt::ApplicationReturnStatus status;
        status = m_app->Initialize();
        if( status != Ipopt::Solve_Succeeded )
        {
            std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
            return;
        }
    }

    virtual ~IpoptInterface() = default;

    using nlp_variable_t = typename IpProblem::nlp_variable_t;
    using nlp_ineq_constraints_t = typename IpProblem::nlp_ineq_constraints_t;
    using nlp_dual_t  = typename IpProblem::nlp_dual_t;
    using parameter_t = typename IpProblem::parameter_t;
    using nlp_settings_t = Ipopt::OptionsList;
    using scalar_t = typename IpProblem::scalar_t;

private:
    scalar_t m_primal_norm{0}, m_dual_norm{0}, m_cost{0}, m_max_violation{0};

public:

    EIGEN_STRONG_INLINE const nlp_settings_t& settings() const noexcept { return *(m_app->Options()); }
    EIGEN_STRONG_INLINE nlp_settings_t& settings() noexcept { return *(m_app->Options()); }

    EIGEN_STRONG_INLINE const Problem& get_problem() const noexcept { return *(m_problem); }
    EIGEN_STRONG_INLINE Problem& get_problem() noexcept { return *(m_problem); }

    EIGEN_STRONG_INLINE const ipopt_info_t& info() const noexcept { return m_info; }
    EIGEN_STRONG_INLINE ipopt_info_t& info() noexcept { return m_info; }

    EIGEN_STRONG_INLINE const nlp_variable_t& primal_solution() const noexcept { return m_problem->m_x; }
    EIGEN_STRONG_INLINE nlp_variable_t& primal_solution() noexcept { return m_problem->m_x; }

    EIGEN_STRONG_INLINE const nlp_dual_t& dual_solution() const noexcept { return m_problem->m_lam; }
    EIGEN_STRONG_INLINE nlp_dual_t& dual_solution() noexcept { return m_problem->m_lam; }

    EIGEN_STRONG_INLINE const nlp_variable_t& lower_bound_x() const noexcept { return m_problem->m_lbx; }
    EIGEN_STRONG_INLINE nlp_variable_t& lower_bound_x() noexcept { return m_problem->m_lbx; }

    EIGEN_STRONG_INLINE const nlp_variable_t& upper_bound_x() const noexcept { return m_problem->m_ubx; }
    EIGEN_STRONG_INLINE nlp_variable_t& upper_bound_x() noexcept { return m_problem->m_ubx; }

    EIGEN_STRONG_INLINE const nlp_ineq_constraints_t& lower_bound_g() const noexcept { return m_problem->m_lbg; }
    EIGEN_STRONG_INLINE nlp_ineq_constraints_t& lower_bound_g() noexcept { return m_problem->m_lbg; }

    EIGEN_STRONG_INLINE const nlp_ineq_constraints_t& upper_bound_g() const noexcept { return m_problem->m_ubg; }
    EIGEN_STRONG_INLINE nlp_ineq_constraints_t& upper_bound_g() noexcept { return m_problem->m_ubg; }

    EIGEN_STRONG_INLINE const parameter_t& parameters() const noexcept { return m_problem->m_p; }
    EIGEN_STRONG_INLINE parameter_t& parameters() noexcept { return m_problem->m_p; }

    EIGEN_STRONG_INLINE const scalar_t primal_norm() const noexcept {return m_primal_norm; }
    EIGEN_STRONG_INLINE const scalar_t dual_norm()   const noexcept {return m_dual_norm;}
    EIGEN_STRONG_INLINE const scalar_t constr_violation() const noexcept {return  m_max_violation;}
    EIGEN_STRONG_INLINE const scalar_t cost() const noexcept {return m_cost;}

    void solve()
    {
        // try to solve the problem
        Ipopt::ApplicationReturnStatus status;
        status = m_app->OptimizeTNLP(m_problem);

        if( status == Ipopt::Solve_Succeeded )
        {
            std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
        }
        else
        {
            std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
        }

        m_info.status = status;
        m_info.iter = (int) m_app->Statistics()->IterationCount();
        Ipopt::Number complementarity, kkt_error, bounds_violation;
        m_app->Statistics()->Infeasibilities(m_dual_norm, m_max_violation, bounds_violation, complementarity, kkt_error);
        m_cost = (scalar_t) m_app->Statistics()->FinalObjective();
    }

    /** solve the NLP with initial guess*/
    /** @bug: make Ipopt about the initial guess */
    void solve(const Eigen::Ref<const nlp_variable_t>& x_guess, const Eigen::Ref<const nlp_dual_t>& lam_guess)
    {
        m_problem->m_x = x_guess;
        m_problem->m_lam = lam_guess;
        this->solve();
    }

};







#endif // IPOPT_INTERFACE_HPP
