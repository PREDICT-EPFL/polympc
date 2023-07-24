// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2022 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "utils/helpers.hpp"
#include "solvers/nlproblem.hpp"
#include "solvers/box_admm.hpp"
#include "solvers/sqp_base.hpp"

// codegen includes
#include "casadi_codegen/cost.h"
#include "casadi_codegen/fcost_gradient.h"
#include "casadi_codegen/fcost_hessian.h"

#include "casadi_codegen/fconstraints.h"
#include "casadi_codegen/fconstraints_jacobian.h"

#include "casadi_codegen/flagrangian.h"
#include "casadi_codegen/flagrangian_gradient.h"
#include "casadi_codegen/flagrangian_hessian.h"

// standart
#include <iomanip>
#include <functional>

// test
#include "gtest/gtest.h"

static void allocate_sparse_matrix_codegen(casadi_int& rows, casadi_int& cols, casadi_int& nnz, int* &outter, int* &inner,
                                           casadi_real** &values, const std::function<const casadi_int*(casadi_int)>& sparsity_info)
{
    const casadi_int* sparsity = sparsity_info(0);
    if(sparsity != nullptr)
    {
        rows = sparsity[0];
        cols = sparsity[1];

        nnz = (sparsity[2] == 0) ? sparsity[2 + cols] : cols * rows;
    }

    values = new casadi_real*[1];
    values[0] = new casadi_real[nnz];
    outter = new int[cols + 1];
    inner  = new int[nnz];

    for(size_t i = 0; i < cols + 1; i++)
        outter[i] = sparsity[2 + i];

    for(size_t i = 0; i < nnz; i++)
        inner[i] = sparsity[3 + cols + i];
}



// Constrained Rosenbrock Function
POLYMPC_FORWARD_NLP_DECLARATION(/*Name*/ MobileRobotCG, /*NX*/ 55, /*NE*/33, /*NI*/0, /*NP*/0, /*Type*/double);
class MobileRobotCG : public ProblemBase<MobileRobotCG, SPARSE>
{
public:

    casadi_real** cg_x;
    casadi_real** cg_cost;
    casadi_real** cg_constraints;
    casadi_real** cg_gradient;

    // Jacobian
    casadi_real** cg_jacobian_values;
    int*  cg_jac_outter;
    int*  cg_jac_inner;
    casadi_int jac_rows = 0, jac_cols = 0, jac_nnz = 0;

    // Hessian
    casadi_real** cg_hessian_values;
    int* cg_hess_outter;
    int* cg_hess_inner;
    casadi_int hess_rows = 0, hess_cols = 0, hess_nnz = 0;

    // Lagrangian Hessian
    casadi_real** cg_lag_hessian_values;
    int* cg_lag_hess_outter;
    int* cg_lag_hess_inner;
    casadi_int lag_hess_nnz;

    // allocate memory for codegen
    MobileRobotCG()
    {
        // optimisation variable
        cg_x = new casadi_real*[2];
        cg_x[0] = new casadi_real[VAR_SIZE];
        cg_x[1] = new casadi_real[DUAL_SIZE];

        // cost
        cg_cost = new casadi_real*[1];
        cg_cost[0] = new casadi_real[1];

        // equality constraints
        cg_constraints = new casadi_real*[1];
        cg_constraints[0] = new casadi_real[NUM_EQ];

        // gradients
        cg_gradient = new casadi_real*[1];
        cg_gradient[0] = new casadi_real[VAR_SIZE];

        // constraints jacobian
        allocate_sparse_matrix_codegen(jac_rows, jac_cols, jac_nnz, cg_jac_outter, cg_jac_inner,
                                       cg_jacobian_values, fconstraints_jacobian_sparsity_out);

        // cost Hessian
        allocate_sparse_matrix_codegen(hess_rows, hess_cols, hess_nnz, cg_hess_outter, cg_hess_inner,
                                       cg_hessian_values, fcost_hessian_sparsity_out);

        // allocate memory for the Lagrangian Hessian
        allocate_sparse_matrix_codegen(hess_rows, hess_cols, lag_hess_nnz, cg_lag_hess_outter, cg_lag_hess_inner,
                                       cg_lag_hessian_values, flagrangian_hessian_sparsity_out);

    }

    ~MobileRobotCG()
    {
        // clear memory
        delete[] cg_x[0];
        delete[] cg_x[1];
        delete[] cg_x;

        delete[] cg_cost[0];
        delete[] cg_cost;

        // delete constraints
        delete[] cg_constraints[0];
        delete[] cg_constraints;

        // delete Jacobian
        delete[] cg_jac_inner;
        delete[] cg_jac_outter;
        delete[] cg_jacobian_values[0];
        delete[] cg_jacobian_values;

        // delete cost Hessian
        delete[] cg_hess_inner;
        delete[] cg_hess_outter;
        delete[] cg_hessian_values[0];
        delete[] cg_hessian_values;

        // delete Lagrangian Hessian
        delete[] cg_lag_hess_inner;
        delete[] cg_lag_hess_outter;
        delete[] cg_lag_hessian_values[0];
        delete[] cg_lag_hessian_values;
    }

    template<typename T>
    EIGEN_STRONG_INLINE void cost_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p, T& cost) const noexcept
    {
        // cast to the raw vector
        Eigen::Map<nlp_variable_t>(cg_x[0], VAR_SIZE, 1) = x;
        fcost((const casadi_real**)cg_x, cg_cost, nullptr, nullptr, nullptr);

        cost = cg_cost[0][0];
    }

    template<typename T>
    EIGEN_STRONG_INLINE void equality_constraints_impl(const Eigen::Ref<const variable_t<T>>& x, const Eigen::Ref<const static_parameter_t>& p,
                                                       Eigen::Ref<constraint_t<T>> constraint) const noexcept
    {
        Eigen::Map<nlp_variable_t>(cg_x[0], VAR_SIZE, 1) = x;
        fconstraint((const casadi_real**)cg_x, cg_constraints, nullptr, nullptr, nullptr);

        constraint = Eigen::Map<nlp_eq_constraints_t>(cg_constraints[0], NUM_EQ, 1);
    }

    EIGEN_STRONG_INLINE void constraints_linearised_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                         const Eigen::Ref<const static_parameter_t>& p,
                                                         Eigen::Ref<nlp_constraints_t> constraints,
                                                         nlp_jacobian_t& jacobian) noexcept
    {
        Eigen::Map<nlp_variable_t>(cg_x[0], VAR_SIZE, 1) = var;
        fconstraint((const casadi_real**)cg_x, cg_constraints, nullptr, nullptr, nullptr);
        fconstraints_jacobian((const casadi_real**)cg_x, cg_jacobian_values, nullptr, nullptr, nullptr);

        constraints = Eigen::Map<nlp_eq_constraints_t>(cg_constraints[0], NUM_EQ, 1);

        if(jacobian.nonZeros() != jac_nnz)
        {
            jacobian = Eigen::Map<Eigen::SparseMatrix<double>>(jac_rows, jac_cols, jac_nnz,
                                                               cg_jac_outter, cg_jac_inner, cg_jacobian_values[0]);

            jacobian.uncompress();
        }
        else
        {
            std::copy_n(cg_jacobian_values[0], jac_nnz, jacobian.valuePtr());
        }
    }

    EIGEN_STRONG_INLINE void cost_gradient_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                                scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient) noexcept
    {
        Eigen::Map<nlp_variable_t>(cg_x[0], VAR_SIZE, 1) = var;
        fcost((const casadi_real**)cg_x, cg_cost, nullptr, nullptr, nullptr);
        fcost_gradient((const casadi_real**)cg_x, cg_gradient, nullptr, nullptr, nullptr);

        _cost = cg_cost[0][0];
        _cost_gradient = Eigen::Map<nlp_variable_t>(cg_gradient[0], VAR_SIZE, 1);
    }

    //compute cost, gradient and hessian
    EIGEN_STRONG_INLINE void cost_gradient_hessian_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                        const Eigen::Ref<const static_parameter_t>& p,
                                                        scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient,
                                                        nlp_hessian_t& hessian) noexcept
    {
        Eigen::Map<nlp_variable_t>(cg_x[0], VAR_SIZE, 1) = var;
        fcost((const casadi_real**)cg_x, cg_cost, nullptr, nullptr, nullptr);
        fcost_gradient((const casadi_real**)cg_x, cg_gradient, nullptr, nullptr, nullptr);
        fcost_hessian((const casadi_real**)cg_x, cg_hessian_values, nullptr, nullptr, nullptr);

        _cost = cg_cost[0][0];
        _cost_gradient = Eigen::Map<nlp_variable_t>(cg_gradient[0], VAR_SIZE, 1);

        if(hessian.nonZeros() != hess_nnz)
        {
            hessian = Eigen::Map<Eigen::SparseMatrix<double>>(hess_rows, hess_cols, hess_nnz,
                                                              cg_hess_outter, cg_hess_inner, cg_hessian_values[0]);
            hessian.uncompress();
        }
        else
        {
            std::copy_n(cg_hessian_values[0], hess_nnz, hessian.valuePtr());
        }
    }

    EIGEN_STRONG_INLINE void lagrangian_impl(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p,
                                             const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian) const noexcept
    {
        Eigen::Map<nlp_variable_t>(cg_x[0], VAR_SIZE, 1) = var;
        Eigen::Map<nlp_dual_t>(cg_x[1], DUAL_SIZE, 1) = lam;
        flagrangian((const casadi_real**)cg_x, cg_cost, nullptr, nullptr, nullptr);

        _lagrangian = cg_cost[0][0];
    }

    EIGEN_STRONG_INLINE void lagrangian_gradient_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                      const Eigen::Ref<const static_parameter_t>& p,
                                                      const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian,
                                                      Eigen::Ref<nlp_variable_t> _lag_gradient) noexcept
    {
        Eigen::Map<nlp_variable_t>(cg_x[0], VAR_SIZE, 1) = var;
        Eigen::Map<nlp_dual_t>(cg_x[1], DUAL_SIZE, 1) = lam;

        flagrangian((const casadi_real**)cg_x, cg_cost, nullptr, nullptr, nullptr);
        flagrangian_gradient((const casadi_real**)cg_x, cg_gradient, nullptr, nullptr, nullptr);

        _lagrangian = cg_cost[0][0];
        _lag_gradient = Eigen::Map<nlp_variable_t>(cg_gradient[0], VAR_SIZE, 1);
        _lag_gradient.noalias() += lam.tail<VAR_SIZE>();
    }

    EIGEN_STRONG_INLINE void lagrangian_gradient_hessian_impl(const Eigen::Ref<const nlp_variable_t> &var, const Eigen::Ref<const static_parameter_t> &p,
                                                              const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian,
                                                              Eigen::Ref<nlp_variable_t> lag_gradient, MATRIX_REF(is_sparse,nlp_hessian_t) lag_hessian,
                                                              Eigen::Ref<nlp_variable_t> cost_gradient,
                                                              Eigen::Ref<nlp_constraints_t> g, MATRIX_REF(is_sparse,nlp_jacobian_t) jac_g,
                                                              const scalar_t cost_scale) noexcept
    {
        Eigen::Map<nlp_variable_t>(cg_x[0], VAR_SIZE, 1) = var;
        Eigen::Map<nlp_dual_t>(cg_x[1], DUAL_SIZE, 1) = lam;

        // Lagrangian
        flagrangian((const casadi_real**)cg_x, cg_cost, nullptr, nullptr, nullptr);
        flagrangian_gradient((const casadi_real**)cg_x, cg_gradient, nullptr, nullptr, nullptr);
        flagrangian_hessian((const casadi_real**)cg_x, cg_lag_hessian_values, nullptr, nullptr, nullptr);

        _lagrangian = cg_cost[0][0];
        lag_gradient = Eigen::Map<nlp_variable_t>(cg_gradient[0], VAR_SIZE, 1);
        lag_gradient.noalias() += lam.tail<VAR_SIZE>();

        if(lag_hessian.nonZeros() != lag_hess_nnz)
        {
            lag_hessian = Eigen::Map<Eigen::SparseMatrix<double>>(hess_rows, hess_cols, lag_hess_nnz,
                                                                  cg_lag_hess_outter, cg_lag_hess_inner, cg_lag_hessian_values[0]);
            lag_hessian.uncompress();
        }
        else
        {
            // just copy the values
            std::copy_n(cg_lag_hessian_values[0], lag_hess_nnz, lag_hessian.valuePtr());
        }

        // cost
        fcost_gradient((const casadi_real**)cg_x, cg_gradient, nullptr, nullptr, nullptr);
        cost_gradient = Eigen::Map<nlp_variable_t>(cg_gradient[0], VAR_SIZE, 1);

        // constraints
        fconstraint((const casadi_real**)cg_x, cg_constraints, nullptr, nullptr, nullptr);
        fconstraints_jacobian((const casadi_real**)cg_x, cg_jacobian_values, nullptr, nullptr, nullptr);

        g = Eigen::Map<nlp_eq_constraints_t>(cg_constraints[0], NUM_EQ, 1);

        if(jac_g.nonZeros() != jac_nnz)
        {
            jac_g = Eigen::Map<Eigen::SparseMatrix<double>>(jac_rows, jac_cols, jac_nnz,
                                                            cg_jac_outter, cg_jac_inner, cg_jacobian_values[0]);

            jac_g.uncompress();
        }
        else
        {
            std::copy_n(cg_jacobian_values[0], jac_nnz, jac_g.valuePtr());
        }
    }

};


/** create solver */
template<typename Problem, typename QPSolver> class Solver;
template<typename Problem, typename QPSolver = boxADMM<Problem::VAR_SIZE, Problem::NUM_EQ + Problem::NUM_INEQ,
                                               typename Problem::scalar_t, Problem::MATRIXFMT,
                                               linear_solver_traits<MobileRobotCG::MATRIXFMT>::default_solver>>
class Solver : public SQPBase<Solver<Problem, QPSolver>, Problem, QPSolver>
{
public:
    using Base = SQPBase<Solver<Problem, QPSolver>, Problem, QPSolver>;
    using typename Base::scalar_t;
    using typename Base::nlp_variable_t;
    using typename Base::nlp_hessian_t;
    using typename Base::nlp_jacobian_t;
    using typename Base::nlp_dual_t;
    using typename Base::parameter_t;
    using typename Base::nlp_constraints_t;


    /** change Hessian update algorithm to the one provided by ContinuousOCP*/
    EIGEN_STRONG_INLINE void hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t>& x_step,
                                                 const Eigen::Ref<const nlp_variable_t>& grad_step) noexcept
    {
        this->problem.hessian_update_impl(hessian, x_step, grad_step);
    }

    EIGEN_STRONG_INLINE void hessian_regularisation_dense_impl(Eigen::Ref<nlp_hessian_t> lag_hessian) noexcept
    {
        const int n = this->m_H.rows();
        /**Regularize by the estimation of the minimum negative eigen value--does not work with inexact Hessian update(matrix is already PSD)*/
        scalar_t aii, ri;
        for (int i = 0; i < n; i++)
        {
            aii = lag_hessian(i,i);
            ri  = (lag_hessian.col(i).cwiseAbs()).sum() - abs(aii); // The hessian is symmetric, Gershgorin discs from rows or columns are equal

            if (aii - ri <= 0) {lag_hessian(i,i) += (ri - aii) + scalar_t(0.01);} //All Greshgorin discs are in the positive half

        }
    }


//    EIGEN_STRONG_INLINE void hessian_regularisation_sparse_impl(nlp_hessian_t& lag_hessian) noexcept
//    {
//        const int n = this->m_H.rows(); //132=m_H.toDense().rows()
//        /**Regularize by the estimation of the minimum negative eigen value*/
//        scalar_t aii, ri;
//        for (int i = 0; i < n; i++)
//        {
//            aii = lag_hessian.coeffRef(i, i);
//            ri = (lag_hessian.col(i).cwiseAbs()).sum() - abs(aii); // The hessian is symmetric, Gershgorin discs from rows or columns are equal

//            if (aii - ri <= 0)
//                lag_hessian.coeffRef(i, i) += (ri - aii) + 0.01;//All Gershgorin discs are in the positive half 0.001
//        }
//    }


    /** for this problem it turned out that exact linearisation not only converges faster but also with a lower computation cost per iteration *
     *
     * So we tell the solver to use the exact linearisation here to update the Hessian
     *
     */
    EIGEN_STRONG_INLINE void update_linearisation_dense_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                                             const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                                                             Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                             Eigen::Ref<nlp_jacobian_t> A,Eigen::Ref<nlp_constraints_t> b) noexcept
    {
        this->linearisation_dense_impl(x, p, lam, cost_grad, lag_hessian, A, b);
        polympc::ignore_unused_var(x_step);
    }

    EIGEN_STRONG_INLINE void update_linearisation_sparse_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                                             const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                                                             Eigen::Ref<nlp_variable_t> cost_grad, nlp_hessian_t& lag_hessian,
                                                             nlp_jacobian_t& A, Eigen::Ref<nlp_constraints_t> b) noexcept
    {
        this->linearisation_sparse_impl(x, p, lam, cost_grad, lag_hessian, A, b);
        polympc::ignore_unused_var(x_step);
    }
};


TEST(SQPTestCase, TestCodegenRobotProblem)
{
    /** Test the SQP solver */
    Solver<MobileRobotCG> solver;
    solver.settings().max_iter = 10;
    solver.settings().line_search_max_iter = 10;
    solver.qp_settings().max_iter = 1000;
    Eigen::Matrix<MobileRobotCG::scalar_t, 3, 1> init_cond; init_cond << 0.5, 0.5, 0.5;
    Eigen::Matrix<MobileRobotCG::scalar_t, 2, 1> ub; ub <<  1.5,  0.75;
    Eigen::Matrix<MobileRobotCG::scalar_t, 2, 1> lb; lb << -1.5, -0.75;

    solver.upper_bound_x().tail(22) = ub.replicate(11, 1);
    solver.lower_bound_x().tail(22) = lb.replicate(11, 1);

    solver.upper_bound_x().segment(30, 3) = init_cond;
    solver.lower_bound_x().segment(30, 3) = init_cond;

    //polympc::time_point start = polympc::get_time();
    solver.solve();
    //polympc::time_point stop = polympc::get_time();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

//    std::cout << "Solve status: " << solver.info().status.value << "\n";
//    std::cout << "Num iterations: " << solver.info().iter << "\n";
//    std::cout << "Primal residual: " << solver.primal_norm() << " | dual residual: " << solver.dual_norm()
//              << " | constraints  violation: " << solver.constr_violation() << " | cost: " << solver.cost() <<"\n";
//    std::cout << "Num of QP iter: " << solver.info().qp_solver_iter << "\n";
//    std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";
//    std::cout << "Size of the solver: " << sizeof (solver) << "\n";
//    std::cout << "Solution: " << solver.primal_solution().transpose() << "\n";

    EXPECT_TRUE(solver.info().status.value == sqp_status_t::SOLVED);
    EXPECT_LT(solver.info().iter, solver.settings().max_iter);
}



