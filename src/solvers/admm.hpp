#ifndef ADMM_HPP
#define ADMM_HPP

#include "qp_base.hpp"

template<int N, int M, typename Scalar = double,
         template <typename, int, typename... Args> class LinearSolver = Eigen::LDLT,
         int LinearSolver_UpLo = Eigen::Lower>
class ADMM : public QPBase<ADMM<N, M, Scalar, LinearSolver, LinearSolver_UpLo>, N, M, Scalar, LinearSolver, LinearSolver_UpLo>
{
    using Base = QPBase<ADMM<N, M, Scalar, LinearSolver, LinearSolver_UpLo>, N, M, Scalar, LinearSolver, LinearSolver_UpLo>;
    using qp_var_t        = typename Base::qp_var_t;
    using qp_dual_t       = typename Base::qp_dual_t;
    using qp_constraint_t = typename Base::qp_constraint_t;
    using qp_hessian_t    = typename Base::qp_hessian_t;
    using scalar_t        = typename Base::scalar_t;
    using kkt_mat_t       = typename Base::kkt_mat_t;
    using kkt_vec_t       = typename Base::kkt_vec_t;
    using linear_solver_t = typename Base::linear_solver_t;
    using status_t        = status_t;

public:
    /** ADMM specific */
    qp_var_t m_x_tilde;
    qp_dual_t m_z, m_z_tilde, m_z_prev;
    qp_dual_t m_rho_vec;
    qp_dual_t m_rho_inv_vec;
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
    linear_solver_t linear_solver;

    status_t solve_impl(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
    const Eigen::Ref<const qp_dual_t>& Alb, const Eigen::Ref<qp_dual_t>& Aub) noexcept
    {
        /** setup part */
        kkt_vec_t rhs, x_tilde_nu;
        bool check_termination = false;

        this->m_x.setZero();
        this->m_y.setZero();
        m_z.setZero();

        /** Set QP constraint type */
        this->parse_constraints_bounds(Alb, Aub);

        /** initialize step size (rho) vector */
        rho_vec_update(this->m_settings.rho);

        /** construct KKT matrix (m_K) and compute decomposition */
        construct_kkt_matrix(H, A);
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
            m_z_tilde = m_z_prev + m_rho_inv_vec.cwiseProduct(x_tilde_nu.template tail<M>() - this->m_y);

            /** update x */
            this->m_x.noalias() = this->m_settings.alpha * m_x_tilde + (1 - this->m_settings.alpha) * this->m_x;

            /** update z */
            m_z.noalias() = this->m_settings.alpha * m_z_tilde;
            m_z.noalias() += (1 - this->m_settings.alpha) * m_z_prev + m_rho_inv_vec.cwiseProduct(this->m_y);
            box_projection(m_z, Alb, Aub); // euclidean projection

            /** update y (dual) */
            this->m_y.noalias() += m_rho_vec.cwiseProduct(this->m_settings.alpha * m_z_tilde + (1 - this->m_settings.alpha) * m_z_prev - m_z);

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

            if (this->m_settings.adaptive_rho and iter % this->m_settings.adaptive_rho_interval == 0)
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
        }

        if (iter > this->m_settings.max_iter)
            this->m_info.status = MAX_ITER_EXCEEDED;

        this->m_info.iter = iter;

        return this->m_info.status;
    }


    EIGEN_STRONG_INLINE void construct_kkt_matrix(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_constraint_t>& A) noexcept
    {
        eigen_assert(LinearSolver_UpLo == Eigen::Lower ||
                      LinearSolver_UpLo == (Eigen::Upper|Eigen::Lower));

        m_K.template topLeftCorner<N, N>() = H;
        m_K.template topLeftCorner<N, N>().noalias() += this->m_settings.sigma * qp_hessian_t::Identity(N,N);

        if (LinearSolver_UpLo == (Eigen::Upper|Eigen::Lower))
            m_K.template topRightCorner<N, M>() = A.transpose();

        m_K.template bottomLeftCorner<M, N>() = A;
        m_K.template bottomRightCorner<M, M>() = -1.0 * m_rho_inv_vec.asDiagonal();
    }

    EIGEN_STRONG_INLINE void factorise_kkt_matrix() noexcept
    {
        /** try implace decomposition */
        linear_solver.compute(m_K);
        eigen_assert(linear_solver.info() == Eigen::Success);
    }

    EIGEN_STRONG_INLINE void compute_kkt_rhs(const Eigen::Ref<const qp_var_t>& h, Eigen::Ref<kkt_vec_t> rhs) const noexcept
    {
        rhs.template head<N>() = this->m_settings.sigma * this->m_x - h;
        rhs.template tail<M>() = m_z - m_rho_inv_vec.cwiseProduct(this->m_y);
    }

    EIGEN_STRONG_INLINE void box_projection(Eigen::Ref<qp_dual_t> x, const Eigen::Ref<const qp_dual_t>& lbx,
                                            const Eigen::Ref<const qp_dual_t>& ubx) const noexcept
    {
        x = x.cwiseMax(lbx).cwiseMin(ubx);
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
        m_rho_inv_vec = m_rho_vec.cwiseInverse();
        rho = rho0;
        this->m_info.rho_updates += 1;
    }

    void residuals_update(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h,
                          const Eigen::Ref<const qp_constraint_t>& A) noexcept
    {
        scalar_t norm_Ax, norm_z;
        norm_Ax = (A * this->m_x).template lpNorm<Eigen::Infinity>();
        norm_z = m_z.template lpNorm<Eigen::Infinity>();
        m_max_Ax_z_norm = fmax(norm_Ax, norm_z);

        scalar_t norm_Hx, norm_ATy, norm_h;
        norm_Hx  = (H * this->m_x).template lpNorm<Eigen::Infinity>();
        norm_ATy = (A.transpose() * this->m_y).template lpNorm<Eigen::Infinity>();
        norm_h   = h.template lpNorm<Eigen::Infinity>();
        m_max_Hx_ATy_h_norm = fmax(norm_Hx, fmax(norm_ATy, norm_h));

        this->m_info.res_prim = this->primal_residual(A, this->m_x, m_z);
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
        return (this->m_info.res_prim <= eps_prim() and this->m_info.res_dual <= eps_dual()) ? true : false;
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
        m_K.template bottomRightCorner<M, M>() = -1.0 * m_rho_inv_vec.asDiagonal();
    }

};





#endif // ADMM_HPP
