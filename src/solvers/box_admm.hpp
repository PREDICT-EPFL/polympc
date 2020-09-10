#ifndef BOX_ADMM_HPP
#define BOX_ADMM_HPP

#include "qp_base.hpp"

template<int N, int M, typename Scalar = double,
         template <typename, int, typename... Args> class LinearSolver = Eigen::LDLT,
         int LinearSolver_UpLo = Eigen::Lower>
class boxADMM : public QPBase<boxADMM<N, M, Scalar, LinearSolver, LinearSolver_UpLo>, N, M, Scalar, LinearSolver, LinearSolver_UpLo>
{
    using Base = QPBase<boxADMM<N, M, Scalar, LinearSolver, LinearSolver_UpLo>, N, M, Scalar, LinearSolver, LinearSolver_UpLo>;
    using qp_var_t        = typename Base::qp_var_t;
    using qp_dual_t       = typename Base::qp_dual_t;
    using qp_constraint_t = typename Base::qp_constraint_t;
    using qp_hessian_t    = typename Base::qp_hessian_t;
    using scalar_t        = typename Base::scalar_t;
    using kkt_mat_t       = typename Base::kkt_mat_t;
    using kkt_vec_t       = typename Base::kkt_vec_t;
    using linear_solver_t = typename Base::linear_solver_t;

public:
    using status_t        = status_t;
    /** ADMM specific */
    qp_var_t m_x_tilde, m_y_box;
    qp_var_t m_q, m_q_prev;
    qp_dual_t m_z, m_z_tilde, m_z_prev;
    qp_dual_t m_rho_vec, m_rho_inv_vec;
    qp_var_t  m_rho_vec_box, m_rho_vec_box_inv, m_rho_vec_box_prev;
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

    status_t solve_box_impl(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
                            const Eigen::Ref<const qp_dual_t>& Alb, const Eigen::Ref<const qp_dual_t>& Aub,
                            const Eigen::Ref<const qp_var_t>& xlb, const Eigen::Ref<const qp_var_t>& xub) noexcept
    {
        return solve_box_impl(H, h, A, Alb, Aub, xlb, xub, qp_var_t::Zero(N,1), qp_dual_t::Zero(M,1));
    }


    status_t solve_box_impl(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
                        const Eigen::Ref<const qp_dual_t>& Alb, const Eigen::Ref<const qp_dual_t>& Aub,
                        const Eigen::Ref<const qp_var_t>& xlb, const Eigen::Ref<const qp_var_t>& xub,
                        const Eigen::Ref<const qp_var_t>& x_guess, const Eigen::Ref<const qp_dual_t>& y_guess) noexcept
    {
        /** setup part */
        kkt_vec_t rhs, x_tilde_nu;
        bool check_termination = false;

        this->m_x = x_guess;
        this->m_y = y_guess; // change guess here
        this->m_z.noalias() = A * x_guess;
        this->m_q = this->m_x;

        /** Set QP constraint type */
        this->parse_constraints_bounds(Alb, Aub, xlb, xub);

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
            m_q_prev = m_q;

            /** update x_tilde z_tilde */
            compute_kkt_rhs(h, rhs);
            x_tilde_nu = linear_solver.solve(rhs);

            m_x_tilde = x_tilde_nu.template head<N>();
            m_z_tilde = m_z_prev + m_rho_inv_vec.cwiseProduct(x_tilde_nu.template tail<M>() - this->m_y);

            /** update x */
            this->m_x.noalias() = this->m_settings.alpha * m_x_tilde;
            this->m_x.noalias() += (1 - this->m_settings.alpha) * this->m_x;

            /** update z */
            m_z.noalias() = this->m_settings.alpha * m_z_tilde;
            m_z.noalias() += (1 - this->m_settings.alpha) * m_z_prev + m_rho_inv_vec.cwiseProduct(this->m_y);
            m_z = m_z.cwiseMax(Alb).cwiseMin(Aub); //box projection

            /** update q */
            m_q.noalias() = this->m_x + m_rho_vec_box_inv.cwiseProduct(this->m_y_box);
            m_q = m_q.cwiseMax(xlb).cwiseMin(xub); // box projection

            /** update y (dual) */
            this->m_y.noalias() += m_rho_vec.cwiseProduct(this->m_settings.alpha * m_z_tilde + (1 - this->m_settings.alpha) * m_z_prev - m_z);

            /** update y_box (dual) */
            //m_y_box.noalias() += m_rho_vec_box.cwiseProduct(this->m_settings.alpha * m_x_tilde + (1 - this->m_settings.alpha) * m_q_prev - m_q);
            m_y_box.noalias() += m_rho_vec_box.cwiseProduct(this->m_x - m_q);

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
                    //std::cout << "the problem is solved! \n";
                    //std::cout << " residuals: " << this->info().res_prim << " : " << this->info().res_dual << "\n";
                    this->m_info.status = SOLVED;
                    break;
                }
            }

            if (this->m_settings.adaptive_rho and iter % this->m_settings.adaptive_rho_interval == 0)
            {
                // state was not yet updated
                if (!check_termination)
                    residuals_update(H, h, A);

                // adjust rho value and refactorise the KKT matrix
                scalar_t new_rho = estimate_rho(rho);
                new_rho = fmax(RHO_MIN, fmin(new_rho, RHO_MAX));
                this->m_info.rho_estimate = new_rho;

                if (new_rho < rho / this->m_settings.adaptive_rho_tolerance or
                    new_rho > rho * this->m_settings.adaptive_rho_tolerance)
                {
                    m_rho_vec_box_prev = m_rho_vec_box;
                    rho_vec_update(new_rho);
                    update_kkt_rho();
                    // Note: KKT Sparsity pattern unchanged by rho update. Only factorize.
                    factorise_kkt_matrix();
                }
            }

            /**
            std::cout << "iter: " << iter << " primal: " << this->m_x.transpose() << "  ";
            std::cout << "dual: "   << this->m_y.transpose() << " " << m_y_box.transpose() << " | "
                      << " residuals: " << this->info().res_prim << " : " << this->info().res_dual << " | "
                      << " rho: " << m_rho_vec.transpose() << " " << m_rho_vec_box.transpose() << "\n";
                      */
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
        m_K.template topLeftCorner<N, N>() += m_rho_vec_box.asDiagonal();

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
        rhs.template head<N>() = this->m_settings.sigma * this->m_x - h + m_rho_vec_box.cwiseProduct(m_q) - m_y_box;
        rhs.template tail<M>() = m_z - m_rho_inv_vec.cwiseProduct(this->m_y);
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

        /** box constraints */
        for (int i = 0; i < qp_var_t::RowsAtCompileTime; i++)
        {
            switch (this->box_constr_type[i])
            {
            case Base::constraint_type::LOOSE_BOUNDS:
                m_rho_vec_box(i) = RHO_MIN;
                break;
            case Base::constraint_type::EQUALITY_CONSTRAINT:
                m_rho_vec_box(i) = RHO_EQ_FACTOR * rho0;
                break;
            case Base::constraint_type::INEQUALITY_CONSTRAINT: /* fall through */
            default:
                m_rho_vec_box(i) = rho0;
            };
        }
        m_rho_vec_box_inv = m_rho_vec_box.cwiseInverse();

        this->m_info.rho_updates += 1;
    }

    void residuals_update(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h,
                          const Eigen::Ref<const qp_constraint_t>& A) noexcept
    {
        scalar_t norm_Ax, norm_z;
        norm_Ax = (A * this->m_x).template lpNorm<Eigen::Infinity>();
        norm_z = m_z.template lpNorm<Eigen::Infinity>();
        m_max_Ax_z_norm = fmax(norm_Ax, norm_z);

        scalar_t norm_Hx, norm_ATy, norm_h, norm_y_box;
        norm_Hx  = (H * this->m_x).template lpNorm<Eigen::Infinity>();
        norm_ATy = (A.transpose() * this->m_y).template lpNorm<Eigen::Infinity>();
        norm_h   = h.template lpNorm<Eigen::Infinity>();
        norm_y_box = m_y_box.template lpNorm<Eigen::Infinity>();
        m_max_Hx_ATy_h_norm = fmax(norm_Hx, fmax(norm_ATy, fmax(norm_h, norm_y_box)));

        this->m_info.res_prim = this->primal_residual(A, this->m_x, m_z) + (this->m_x - m_q).template lpNorm<Eigen::Infinity>();
        this->m_info.res_dual = this->dual_residual_(H, h, A, this->m_x, this->m_y, m_y_box);
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
        m_K.template topLeftCorner<N, N>() += (m_rho_vec_box - m_rho_vec_box_prev).asDiagonal();
        m_K.template bottomRightCorner<M, M>() = -1.0 * m_rho_inv_vec.asDiagonal();
    }

    /** dual residual estimation */
    EIGEN_STRONG_INLINE scalar_t dual_residual_(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h,
                                               const Eigen::Ref<const qp_constraint_t>& A, const Eigen::Ref<const qp_var_t>& x,
                                               const Eigen::Ref<const qp_dual_t>& y, const Eigen::Ref<const qp_var_t>& y_box) const noexcept
    {
        return (H * x + h + A.transpose() * y + y_box).template lpNorm<Eigen::Infinity>();
    }

};





#endif // BOX_ADMM_HPP
