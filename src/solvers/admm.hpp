#ifndef ADMM_HPP
#define ADMM_HPP

#include "qp_base.hpp"

template<int N, int M, typename Scalar = double, int MatrixType = DENSE,
         template <typename, int, typename... Args> class LinearSolver = linear_solver_traits<DENSE>::default_solver,
         int LinearSolver_UpLo = Eigen::Lower, typename ...Args>
class ADMM : public QPBase<ADMM<N, M, Scalar, MatrixType, LinearSolver, LinearSolver_UpLo>, N, M, Scalar, MatrixType, LinearSolver, LinearSolver_UpLo>
{
    using Base = QPBase<ADMM<N, M, Scalar, MatrixType, LinearSolver, LinearSolver_UpLo>, N, M, Scalar, MatrixType, LinearSolver, LinearSolver_UpLo>;

public:
    using scalar_t        = typename Base::scalar_t;
    using qp_var_t        = typename Base::qp_var_t;
    using qp_dual_t       = typename Base::qp_dual_t;
    using qp_dual_a_t     = typename Base::qp_dual_a_t;
    using qp_constraint_t = typename Base::qp_constraint_t;
    using qp_hessian_t    = typename Base::qp_hessian_t;
    /** specific for OSQP splitting */


    using kkt_mat_t         = typename std::conditional<MatrixType == SPARSE, Eigen::SparseMatrix<scalar_t>,
                              typename dense_matrix_type_selector<scalar_t, 2 * N + M, 2 * N + M>::type>::type;
    using kkt_vec_t         = typename dense_matrix_type_selector<scalar_t, 2 * N + M, 1>::type;
    using linear_solver_t   = LinearSolver<kkt_mat_t, LinearSolver_UpLo, Args...>; //typename Base::linear_solver_t;
    using admm_dual_t       = typename dense_matrix_type_selector<scalar_t, N + M, 1>::type;
    using admm_constraint_t = typename std::conditional<MatrixType == SPARSE, Eigen::SparseMatrix<scalar_t>,
                              typename dense_matrix_type_selector<scalar_t, N + M, N>::type>::type;

    template<int MatrixFMT = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<MatrixFMT == DENSE>::type allocate_kkt_matrix() noexcept
    {
        if(m_K.RowsAtCompileTime == Eigen::Dynamic) m_K = kkt_mat_t::Zero(2 * N + M, 2 * N + M);
    }

    /** no pre-allocate needed for sparse KKT system */
    template<int MatrixFMT = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<MatrixFMT == SPARSE>::type allocate_kkt_matrix() const noexcept
    {}

    template<int MatrixFMT = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<MatrixFMT == DENSE>::type allocate_jacobian() noexcept
    {
        if(m_A.RowsAtCompileTime == Eigen::Dynamic) m_A = admm_constraint_t::Zero(N + M, N);
    }

    template<int MatrixFMT = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<MatrixFMT == SPARSE>::type allocate_jacobian() const noexcept
    {}

public:
    ADMM() : Base()
    {
        /** intialise some variables */
        m_rho_vec     = admm_dual_t::Constant(this->m_settings.rho);
        m_rho_inv_vec = admm_dual_t::Constant(scalar_t(1/this->m_settings.rho));

        /** allocate space for the KKT matrix if necessary */
        allocate_kkt_matrix();

        /** allocate memory for Jacobian */
        allocate_jacobian();
    }

    ~ADMM() = default;

    /** ADMM specific */
    qp_var_t m_x_tilde;
    admm_dual_t m_z, m_z_tilde, m_z_prev;
    admm_dual_t m_rho_vec, m_rho_inv_vec;
    scalar_t rho;

    int iter{0};
    scalar_t res_prim;
    scalar_t res_dual;

    /** ADMM specific */
    static constexpr scalar_t RHO_MIN = 1e-6;
    static constexpr scalar_t RHO_MAX = 1e+6;
    static constexpr scalar_t RHO_TOL = 1e-4;
    static constexpr scalar_t RHO_EQ_FACTOR = 1e+3;

    /** ADMM specific */
    scalar_t m_max_Ax_z_norm;
    scalar_t m_max_Hx_ATy_h_norm;

    /** solver part */
    kkt_mat_t m_K;
    admm_constraint_t m_A;
    linear_solver_t linear_solver;

    Eigen::VectorXi _kkt_mat_nnz, _A_mat_nnz;

    status_t solve_impl(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
                        const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub,
                        const Eigen::Ref<const qp_var_t>& xl, const Eigen::Ref<const qp_var_t>& xu) noexcept
    {
        return solve_impl(H, h, A, Alb, Aub, xl, xu, qp_var_t::Zero(N,1), qp_dual_t::Zero(M+N,1));
    }


    status_t solve_impl(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
                        const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub,
                        const Eigen::Ref<const qp_var_t>& xl, const Eigen::Ref<const qp_var_t>& xu,
                        const Eigen::Ref<const qp_var_t>& x_guess, const Eigen::Ref<const qp_dual_t>& y_guess) noexcept
    {
        /** setup part */
        kkt_vec_t rhs, x_tilde_nu;
        bool check_termination = false;

        this->m_x = x_guess;
        this->m_y = y_guess;

        /** @bug : create Sparse m_A matrix */
        construct_A(A);
        this->m_z.noalias() = m_A * x_guess;

        /** Set QP constraint type */
        this->parse_constraints_bounds(Alb, Aub, xl, xu);

        /** initialize step size (rho) vector */
        rho_vec_update(this->m_settings.rho);

        /** construct KKT matrix (m_K) and compute decomposition */
        construct_kkt_matrix(H, m_A);
        factorise_kkt_matrix();

        this->m_info.status = UNSOLVED;

        /** run ADMM iterations */
        for (iter = 1; iter <= this->m_settings.max_iter; iter++)
        {
            m_z_prev = m_z;

            /** update x_tilde z_tilde */
            compute_kkt_rhs(h, rhs);
            x_tilde_nu = linear_solver.solve(rhs);

            m_x_tilde = x_tilde_nu.template head<N>();
            m_z_tilde = m_z_prev + m_rho_inv_vec.cwiseProduct(x_tilde_nu.template tail<M + N>() - this->m_y);

            /** update x */
            this->m_x.noalias() = this->m_settings.alpha * m_x_tilde + (1 - this->m_settings.alpha) * this->m_x;

            /** update z */
            m_z.noalias() = this->m_settings.alpha * m_z_tilde;
            m_z.noalias() += (1 - this->m_settings.alpha) * m_z_prev + m_rho_inv_vec.cwiseProduct(this->m_y);
            box_projection(m_z, Alb, Aub, xl, xu); // euclidean projection

            /** update y (dual) */
            this->m_y.noalias() += m_rho_vec.cwiseProduct(this->m_settings.alpha * m_z_tilde +
                                                                             (1 - this->m_settings.alpha) * m_z_prev - m_z);

            if (this->m_settings.check_termination != 0 && iter % this->m_settings.check_termination == 0)
                check_termination = true;
            else
                check_termination = false;

            /** check convergence */
            if (check_termination)
            {
                residuals_update(H, h, A);
                if (termination_criteria())
                {
                    this->m_info.status = SOLVED;
                    break;
                }
            }

            if (this->m_settings.adaptive_rho && iter % this->m_settings.adaptive_rho_interval == 0)
            {
                // state was not yet updated
                if (!check_termination)
                    residuals_update(H, h, A);

                /** adjust rho value and refactorise the KKT matrix */
                scalar_t new_rho = estimate_rho(rho);
                new_rho = fmax(RHO_MIN, fmin(new_rho, RHO_MAX));
                this->m_info.rho_estimate = new_rho;

                if (new_rho < rho / this->m_settings.adaptive_rho_tolerance or
                    new_rho > rho * this->m_settings.adaptive_rho_tolerance)
                {
                    rho_vec_update(new_rho);
                    update_kkt_rho();
                    /* Note: KKT Sparsity pattern unchanged by rho update. Only factorize. */
                    factorise_kkt_matrix();
                }
            }
            /**
            std::cout << "admm iter: " << iter << " | " << this->m_x.transpose() << " | " << this->m_y.transpose() <<
                         " | " << this->m_info.res_prim << " | " << this->m_info.res_dual << " | " << m_rho_vec.transpose() << "\n";
            */
        }


        if (iter > this->m_settings.max_iter)
            this->m_info.status = MAX_ITER_EXCEEDED;

        this->m_info.iter = iter;

        return this->m_info.status;
    }

    /** construct extended A matrix: Ae = [A E] */
    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
    construct_A(const Eigen::Ref<const qp_constraint_t>& A) noexcept
    {
        m_A.template block<M, N>(0,0) = A;
        m_A.template block<N,N>(M,0).setIdentity();
    }

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
            _A_mat_nnz = Eigen::VectorXi::Constant(N, 1); // allocate box constraints
            std::transform(_A_mat_nnz.data(), _A_mat_nnz.data() + N, A.innerNonZeroPtr(), _A_mat_nnz.data(), std::plus<scalar_t>());
            // reserve the memory
            m_A.reserve(_A_mat_nnz);
            block_insert_sparse(m_A, 0, 0, A);  //insert A matrix

            for(Eigen::Index i = 0; i < N; ++i)
                m_A.coeffRef(i + M, i) = scalar_t(1); // insert identity matrix
        }
    }


    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
    construct_kkt_matrix(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const admm_constraint_t>& A) noexcept
    {
        eigen_assert(LinearSolver_UpLo == Eigen::Lower ||
                      LinearSolver_UpLo == (Eigen::Upper|Eigen::Lower));

        m_K.template topLeftCorner<N, N>() = H;
        m_K.template topLeftCorner<N, N>().diagonal() += qp_var_t::Constant(N, this->m_settings.sigma);

        if (LinearSolver_UpLo == (Eigen::Upper|Eigen::Lower))
            m_K.template topRightCorner<N, M + N>() = A.transpose();

        m_K.template bottomLeftCorner<M + N, N>() = A;
        m_K.template bottomRightCorner<M + N, M + N>() = scalar_t(-1) * m_rho_inv_vec.asDiagonal();
    }

    /** sparse implementation */
    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    construct_kkt_matrix(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const admm_constraint_t>& A) noexcept
    {
        if(this->settings().reuse_pattern)
            construct_kkt_matrix_same_pattern(H,A);
        else
        {
            construct_kkt_matrix_sparse(H, A);
            linear_solver.analyzePattern(m_K);
        }
    }

    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    construct_kkt_matrix_sparse(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const admm_constraint_t>& A) noexcept
    {
        /** worst case scenario */
        const int kkt_size = 2 * N + M;
        m_K.resize(kkt_size, kkt_size);
        /** estimate number of nonzeros */
        _kkt_mat_nnz = Eigen::VectorXi::Constant(kkt_size, 1);
        // add nonzeros from H
        std::transform(_kkt_mat_nnz.data(), _kkt_mat_nnz.data() + N, H.innerNonZeroPtr(), _kkt_mat_nnz.data(), std::plus<scalar_t>());
        // add nonzeros from A
        std::transform(_kkt_mat_nnz.data(), _kkt_mat_nnz.data() + N, A.innerNonZeroPtr(), _kkt_mat_nnz.data(), std::plus<scalar_t>());

        // reserve the space and insert values from H and A
        if(LinearSolver_UpLo == (Eigen::Upper|Eigen::Lower))
        {
            // reserve more memory
            _estimate_nnz_in_row(_kkt_mat_nnz, A);

            m_K.reserve(_kkt_mat_nnz);
            block_insert_sparse(m_K, 0, 0, H);
            block_insert_sparse(m_K, N, 0, A);
            block_insert_sparse(m_K, 0, N, A.transpose());
        }
        else
        {
            m_K.reserve(_kkt_mat_nnz);
            block_insert_sparse(m_K, 0, 0, H);
            block_insert_sparse(m_K, N, 0, A);
        }

        // add diagonal blocks*/
        for(Eigen::Index i = 0; i < N; ++i)
            m_K.coeffRef(i, i) += this->m_settings.sigma;
        for(Eigen::Index i = 0; i < N + M; ++i)
            m_K.coeffRef(N + i, N + i) = -m_rho_inv_vec(i);
    }

    /** make faster update if sparsity pattern has not changed */
    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    construct_kkt_matrix_same_pattern(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const admm_constraint_t>& A) noexcept
    {
        /** just copy content of nonzero vectors*/
        // copy H and A blocks
        for(Eigen::Index k = 0; k < N; ++k)
        {
            std::copy_n(H.valuePtr() + H.outerIndexPtr()[k], H.innerNonZeroPtr()[k], m_K.valuePtr() + m_K.outerIndexPtr()[k]);
            std::copy_n(A.valuePtr() + A.outerIndexPtr()[k], A.innerNonZeroPtr()[k], m_K.valuePtr() + m_K.outerIndexPtr()[k] + H.innerNonZeroPtr()[k]);
        }
        // reserve the space and insert values from H and A
        if(LinearSolver_UpLo == (Eigen::Upper|Eigen::Lower))
            block_set_sparse(m_K, N, 0, A.transpose());


        // add diagonal blocks*/
        m_K.diagonal(). template head<N>() += qp_var_t::Constant(this->m_settings.sigma);
        m_K.diagonal(). template tail<M + N>() = -m_rho_inv_vec;
    }

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

    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    block_set_sparse(Eigen::SparseMatrix<scalar_t>& dst, const Eigen::Index &row_offset,
                     const Eigen::Index &col_offset, const Eigen::SparseMatrix<scalar_t>& src) const noexcept
    {
        // assumes enough spase is allocated in the dst matrix
        for(Eigen::Index k = 0; k < src.outerSize(); ++k)
            for (typename Eigen::SparseMatrix<scalar_t>::InnerIterator it(src, k); it; ++it)
                dst.insert(row_offset + it.row(), col_offset + it.col()) = it.value();
    }

    /** workaround function to estimate number of nonzeros */
    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
    _estimate_nnz_in_row(Eigen::Ref<Eigen::VectorXi> nnz, const qp_constraint_t& A) const noexcept
    {
        for (Eigen::Index k = 0; k < A.outerSize(); ++k)
            for(typename qp_constraint_t::InnerIterator it(A, k); it; ++it)
                nnz(it.row() + N) += scalar_t(1);
    }


    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type factorise_kkt_matrix() noexcept
    {
        /** try implace decomposition */
        linear_solver.compute(m_K);
        eigen_assert(linear_solver.info() == Eigen::Success);
    }

    template<int T = MatrixType>
    EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type factorise_kkt_matrix() noexcept
    {
        /** try implace decomposition */
        linear_solver.factorize(m_K);
        eigen_assert(linear_solver.info() == Eigen::Success);
    }


    EIGEN_STRONG_INLINE void compute_kkt_rhs(const Eigen::Ref<const qp_var_t>& h, Eigen::Ref<kkt_vec_t> rhs) const noexcept
    {
        rhs.template head<N>() = this->m_settings.sigma * this->m_x - h;
        rhs.template tail<M + N>() = m_z - m_rho_inv_vec.cwiseProduct(this->m_y);
    }

    EIGEN_STRONG_INLINE void box_projection(Eigen::Ref<admm_dual_t> x, const Eigen::Ref<const qp_dual_a_t>& lba,
                                            const Eigen::Ref<const qp_dual_a_t>& uba,
                                            const Eigen::Ref<const qp_var_t>& lbx, const Eigen::Ref<const qp_var_t>& ubx) const noexcept
    {
        x.template head<M>() = x.template head<M>().cwiseMax(lba).cwiseMin(uba);
        x.template tail<N>() = x.template tail<N>().cwiseMax(lbx).cwiseMin(ubx);
    }

    EIGEN_STRONG_INLINE void rho_vec_update(const scalar_t& rho0) noexcept
    {
        for (int i = 0; i < qp_dual_t::RowsAtCompileTime; i++)
        {
            switch (this->constr_type[i])
            {
            case Base::constraint_type::LOOSE_BOUNDS:
                m_rho_vec(i) = RHO_MIN;
                break;
            case Base::constraint_type::EQUALITY_CONSTRAINT:
                m_rho_vec(i) = RHO_EQ_FACTOR * rho0;
                break;
            case Base::constraint_type::INEQUALITY_CONSTRAINT: /* fall through */
            default:
                m_rho_vec(i) = rho0;
            };
        }

        /** box constraints */
        for (int i = 0; i < qp_var_t::RowsAtCompileTime; i++)
        {
            switch (this->box_constr_type[i])
            {
            case Base::constraint_type::LOOSE_BOUNDS:
                m_rho_vec(i + M) = RHO_MIN;
                break;
            case Base::constraint_type::EQUALITY_CONSTRAINT:
                m_rho_vec(i + M) = RHO_EQ_FACTOR * rho0;
                break;
            case Base::constraint_type::INEQUALITY_CONSTRAINT: /* fall through */
            default:
                m_rho_vec(i + M) = rho0;
            };
        }

        m_rho_inv_vec = m_rho_vec.cwiseInverse();
        rho = rho0;

        this->m_info.rho_updates += 1;
    }

    void residuals_update(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h,
                          const Eigen::Ref<const qp_constraint_t>& A) noexcept
    {
        scalar_t norm_Ax, norm_z;
        norm_Ax = (A * this->m_x).template lpNorm<Eigen::Infinity>();
        norm_Ax = fmax(norm_Ax, this->m_x.template tail<N>().template lpNorm<Eigen::Infinity>());
        norm_z = m_z.template lpNorm<Eigen::Infinity>();
        m_max_Ax_z_norm = fmax(norm_Ax, norm_z);

        scalar_t norm_Hx, norm_ATy, norm_h, norm_y_box;
        norm_Hx  = (H * this->m_x).template lpNorm<Eigen::Infinity>();
        norm_ATy = (this->m_y.template head<M>().transpose() * A).template lpNorm<Eigen::Infinity>();
        norm_h   = h.template lpNorm<Eigen::Infinity>();
        norm_y_box = this->m_y.template tail<N>().template lpNorm<Eigen::Infinity>();
        m_max_Hx_ATy_h_norm = fmax(norm_Hx, fmax(norm_ATy, fmax(norm_h, norm_y_box)));

        this->m_info.res_prim = this->primal_residual(A, this->m_x, m_z.template head<M>());
        scalar_t box_norm = (this->m_x - m_z.template tail<N>()).template lpNorm<Eigen::Infinity>();
        this->m_info.res_prim = fmax(this->m_info.res_prim, box_norm);

        this->m_info.res_dual = this->dual_residual(H, h, A, this->m_x, this->m_y);
    }

    EIGEN_STRONG_INLINE scalar_t eps_prim() const noexcept
    {
        return this->m_settings.eps_abs + this->m_settings.eps_rel * m_max_Ax_z_norm;
    }

    EIGEN_STRONG_INLINE scalar_t eps_dual() const noexcept
    {
        return this->m_settings.eps_abs + this->m_settings.eps_rel * m_max_Hx_ATy_h_norm;
    }

    EIGEN_STRONG_INLINE bool termination_criteria() const noexcept
    {
        // check residual norms to detect optimality
        return (this->m_info.res_prim <= eps_prim() && this->m_info.res_dual <= eps_dual()) ? true : false;
    }

    EIGEN_STRONG_INLINE scalar_t estimate_rho(const scalar_t& rho0) const noexcept
    {
        scalar_t rp_norm, rd_norm;
        rp_norm = this->m_info.res_prim / (m_max_Ax_z_norm + this->DIV_BY_ZERO_REGUL);
        rd_norm = this->m_info.res_dual / (m_max_Hx_ATy_h_norm + this->DIV_BY_ZERO_REGUL);

        scalar_t rho_new = rho0 * sqrt(rp_norm / (rd_norm + this->DIV_BY_ZERO_REGUL));
        return rho_new;
    }

    EIGEN_STRONG_INLINE void update_kkt_rho() noexcept
    {
        m_K.diagonal(). template tail<N + M>() = -m_rho_inv_vec;
        //m_K.template bottomRightCorner<M + N, M + N>() = -1.0 * m_rho_inv_vec.asDiagonal();
    }


};





#endif // ADMM_HPP
