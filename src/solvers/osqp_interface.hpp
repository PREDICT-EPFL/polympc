// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OSQP_INTERFACE_HPP
#define OSQP_INTERFACE_HPP

#include "solvers/qp_base.hpp"
#include "OsqpEigen/OsqpEigen.h"

namespace polympc {

template<int N, int M, typename Scalar = double, int MatrixType = SPARSE>
class OSQP : public QPBase<OSQP<N, M, Scalar, MatrixType>, N, M, Scalar, MatrixType>
{
    using Base = QPBase<OSQP<N, M, Scalar, MatrixType>, N, M, Scalar, MatrixType>;
    using qp_var_t        = typename Base::qp_var_t;
    using qp_dual_t       = typename Base::qp_dual_t;
    using qp_dual_a_t     = typename Base::qp_dual_a_t;
    using qp_constraint_t = typename Base::qp_constraint_t;
    using qp_hessian_t    = typename Base::qp_hessian_t;
    using scalar_t        = typename Base::scalar_t;

    using admm_constraint_t = typename std::conditional<MatrixType == SPARSE, Eigen::SparseMatrix<scalar_t>,
                              typename dense_matrix_type_selector<scalar_t, N + M, N>::type>::type;

public:
    /** constructor */
    OSQP() : Base()
    {
        EIGEN_STATIC_ASSERT(MatrixType == SPARSE, "OSQP_Interface: OSQP does not support dense matrices \n");
        EIGEN_STATIC_ASSERT((std::is_same<Scalar, c_float>::value == true), "OSQP_Interface: OSQP is set to work with 'double' precision, "
                                                                            "consider passing 'DFLOAT' definition to your compiler \n");
        set_osqp_settings();
        osqp_solver.data()->setNumberOfVariables(N);
        osqp_solver.data()->setNumberOfConstraints(N + M); // N + M
    }
    ~OSQP() = default;

    /** solve */
    status_t solve_impl(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
                        const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub,
                        const Eigen::Ref<const qp_var_t>& xlb, const Eigen::Ref<const qp_var_t>& xub) noexcept
    {
        set_osqp_settings();

        construct_A(A);

        // use Ref
        x_upper_bound. template head<M>() = Aub;
        x_upper_bound. template tail<N>() = xub;
        x_lower_bound. template head<M>() = Alb;
        x_lower_bound. template tail<N>() = xlb;

        if(this->m_settings.warm_start)
        {
            osqp_solver.updateHessianMatrix(H);
            osqp_solver.updateGradient(h);
            osqp_solver.updateLinearConstraintsMatrix(m_A);
            osqp_solver.updateBounds(x_lower_bound, x_upper_bound);

            osqp_solver.solveProblem();
        }
        else
        {
            m_h = h;

            osqp_solver.data()->setHessianMatrix(H);
            osqp_solver.data()->setGradient(m_h);
            osqp_solver.data()->setLinearConstraintsMatrix(m_A);
            osqp_solver.data()->setLowerBound(x_lower_bound);
            osqp_solver.data()->setUpperBound(x_upper_bound);

            osqp_solver.initSolver();
            osqp_solver.solveProblem();
        }

        this->m_x = osqp_solver.getSolution();
        this->m_y = osqp_solver.getDualSolution();

        this->m_info.iter = osqp_solver.workspace()->info->iter;
        this->m_info.res_prim = osqp_solver.workspace()->info->dua_res;
        this->m_info.rho_updates = osqp_solver.workspace()->info->rho_updates;
        this->m_info.rho_estimate = osqp_solver.workspace()->info->rho_estimate;

        // update status
        switch (osqp_solver.workspace()->info->status_val)
        {
            case OSQP_SOLVED : {this->m_info.status = SOLVED; return SOLVED;}
            case OSQP_SOLVED_INACCURATE : {this->m_info.status = SOLVED; return SOLVED;}
            case OSQP_PRIMAL_INFEASIBLE : {this->m_info.status = INFEASIBLE; return INFEASIBLE;}
            case OSQP_DUAL_INFEASIBLE : {this->m_info.status = INFEASIBLE; return INFEASIBLE;}
            case OSQP_MAX_ITER_REACHED : {this->m_info.status = MAX_ITER_EXCEEDED; return MAX_ITER_EXCEEDED;}
            case OSQP_UNSOLVED : {this->m_info.status = UNSOLVED; return UNSOLVED;}
            default : {this->m_info.status = UNSOLVED; return UNSOLVED;}
        }
    }

    status_t solve_impl(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
                        const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub,
                        const Eigen::Ref<const qp_var_t>& xlb, const Eigen::Ref<const qp_var_t>& xub,
                        const Eigen::Ref<const qp_var_t>& x_guess, const Eigen::Ref<const qp_dual_t>& y_guess) noexcept
    {
        set_osqp_settings();
        construct_A(A);

        // use Ref
        x_upper_bound. template head<M>() = Aub;
        x_lower_bound. template head<M>() = Alb;
        x_upper_bound. template tail<N>() = xub;
        x_lower_bound. template tail<N>() = xlb;

        //Eigen::VectorXd h_ = h; // hack start an issue on github
        if(this->m_settings.warm_start)
        {
            osqp_solver.updateHessianMatrix(H);
            osqp_solver.updateGradient(h);
            osqp_solver.updateLinearConstraintsMatrix(m_A);
            osqp_solver.updateBounds(x_lower_bound, x_upper_bound);


            osqp_solver.setWarmStart(x_guess, y_guess);
            osqp_solver.solveProblem();
        }
        else
        {
            m_h = h;

            osqp_solver.data()->setHessianMatrix(H);
            osqp_solver.data()->setGradient(m_h);
            osqp_solver.data()->setLinearConstraintsMatrix(m_A);
            osqp_solver.data()->setLowerBound(x_lower_bound);
            osqp_solver.data()->setUpperBound(x_upper_bound);

            osqp_solver.initSolver();
            osqp_solver.setWarmStart(x_guess, y_guess);
            osqp_solver.solveProblem();
        }

        this->m_x = osqp_solver.getSolution();
        this->m_y = osqp_solver.getDualSolution();

        this->m_info.iter = osqp_solver.workspace()->info->iter;
        this->m_info.res_prim = osqp_solver.workspace()->info->dua_res;
        this->m_info.rho_updates = osqp_solver.workspace()->info->rho_updates;
        this->m_info.rho_estimate = osqp_solver.workspace()->info->rho_estimate;

        // update status
        switch (osqp_solver.workspace()->info->status_val)
        {
            case OSQP_SOLVED : {this->m_info.status = SOLVED; return SOLVED;}
            case OSQP_SOLVED_INACCURATE : {this->m_info.status = SOLVED; return SOLVED;}
            case OSQP_PRIMAL_INFEASIBLE : {this->m_info.status = INFEASIBLE; return INFEASIBLE;}
            case OSQP_DUAL_INFEASIBLE : {this->m_info.status = INFEASIBLE; return INFEASIBLE;}
            case OSQP_MAX_ITER_REACHED : {this->m_info.status = MAX_ITER_EXCEEDED; return MAX_ITER_EXCEEDED;}
            case OSQP_UNSOLVED : {this->m_info.status = UNSOLVED; return UNSOLVED;}
            default : {this->m_info.status = UNSOLVED; return UNSOLVED;}
        }
    }

    const OsqpEigen::Solver& get_solver_data() const noexcept {return osqp_solver;}

private:
    OsqpEigen::Solver osqp_solver;
    OsqpEigen::Settings osqp_settings;
    admm_constraint_t m_A;
    typename dense_matrix_type_selector<scalar_t, N + M, 1>::type x_lower_bound, x_upper_bound;
    typename dense_matrix_type_selector<scalar_t, N, 1>::type m_h;

    /** construct the Jacobian matrix accepted by OSQP */
    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    construct_A(const Eigen::Ref<const qp_constraint_t>& A) noexcept
    {
        if(this->settings().reuse_pattern)
        {
            /** copy the new A block */
            for(Eigen::Index k = 0; k < N; ++k)
                std::copy_n(A.valuePtr() + A.outerIndexPtr()[k], A.innerNonZeroPtr()[k], m_A.valuePtr() + m_A.outerIndexPtr()[k]);
        }
        else
        {
            m_A.resize(N + M, N);
            Eigen::VectorXi _A_mat_nnz = Eigen::VectorXi::Constant(N, 1); // allocate box constraints
            std::transform(_A_mat_nnz.data(), _A_mat_nnz.data() + N, A.innerNonZeroPtr(), _A_mat_nnz.data(), std::plus<scalar_t>());
            // reserve the memory
            m_A.reserve(_A_mat_nnz);
            block_insert_sparse(m_A, 0, 0, A);  //insert A matrix

            for(Eigen::Index i = 0; i < N; ++i)
                m_A.coeffRef(i + M, i) = scalar_t(1); // insert identity matrix
        }
    }

    /** insert a block into a sparse matrix */
    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    block_insert_sparse(Eigen::SparseMatrix<scalar_t>& dst, const Eigen::Index &row_offset,
                        const Eigen::Index &col_offset, const Eigen::SparseMatrix<scalar_t>& src) const noexcept
    {
        // assumes enough spase is allocated in the dst matrix
        for(Eigen::Index k = 0; k < src.outerSize(); ++k)
            for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(src, k); it; ++it)
                dst.insert(row_offset + it.row(), col_offset + it.col()) = it.value();
    }

    void set_osqp_settings() noexcept
    {
        osqp_solver.settings()->setRho(this->m_settings.rho);
        osqp_solver.settings()->setAlpha(this->m_settings.alpha);
        osqp_solver.settings()->setSigma(this->m_settings.sigma);
        osqp_solver.settings()->setVerbosity(this->m_settings.verbose);
        osqp_solver.settings()->setWarmStart(this->m_settings.warm_start);
        osqp_solver.settings()->setAdaptiveRho(this->m_settings.adaptive_rho);
        osqp_solver.settings()->setMaxIteration(this->m_settings.max_iter);
        osqp_solver.settings()->setCheckTermination(this->m_settings.check_termination);
        osqp_solver.settings()->setAbsoluteTolerance(this->m_settings.eps_abs);
        osqp_solver.settings()->setRelativeTolerance(this->m_settings.eps_rel);
        osqp_solver.settings()->setAdaptiveRhoInterval(this->m_settings.adaptive_rho_interval);
        osqp_solver.settings()->setAdaptiveRhoTolerance(this->m_settings.adaptive_rho_tolerance);

        osqp_solver.settings()->setDelta(this->m_settings.delta);
        osqp_solver.settings()->setPolish(this->m_settings.polish);
        osqp_solver.settings()->setScaling(this->m_settings.scaling);
        osqp_solver.settings()->setTimeLimit(this->m_settings.time_limit);
        osqp_solver.settings()->setPolishRefineIter(this->m_settings.polish_refine_iter);
        osqp_solver.settings()->setLinearSystemSolver(this->m_settings.osqp_linear_solver);
        osqp_solver.settings()->setScaledTerimination(this->m_settings.scaled_termination);
        osqp_solver.settings()->setAdaptiveRhoFraction(this->m_settings.adaptive_rho_fraction);
        osqp_solver.settings()->setPrimalInfeasibilityTolerance(this->m_settings.eps_prim_inf);
        osqp_solver.settings()->setDualInfeasibilityTolerance(this->m_settings.eps_dual_inf);
    }

};

} // polympc namespace

#endif
