#ifndef QP_BASE_HPP
#define QP_BASE_HPP

#include "Eigen/Dense"
#include <iostream>

template <typename Scalar>
struct qp_sover_settings_t {
    Scalar rho = 1e-1;          /**< ADMM rho step, 0 < rho */
    Scalar sigma = 1e-6;        /**< ADMM sigma step, 0 < sigma, (small) */
    Scalar alpha = 1.0;         /**< ADMM overrelaxation parameter, 0 < alpha < 2,
                                     values in [1.5, 1.8] give good results (empirically) */
    Scalar eps_rel = 1e-3;      /**< Relative tolerance for termination, 0 < eps_rel */
    Scalar eps_abs = 1e-3;      /**< Absolute tolerance for termination, 0 < eps_abs */
    int max_iter = 1000;        /**< Maximal number of iteration, 0 < max_iter */
    int check_termination = 25; /**< Check termination after every Nth iteration, 0 (disabled) or 0 < check_termination */
    bool warm_start = false;    /**< Warm start solver, reuses previous x,z,y */
    bool adaptive_rho = false;  /**< Adapt rho to optimal estimate */
    Scalar adaptive_rho_tolerance = 5;  /**< Minimal for rho update factor, 1 < adaptive_rho_tolerance */
    int adaptive_rho_interval = 25; /**< change rho every Nth iteration, 0 < adaptive_rho_interval,
                                         set equal to check_termination to save computation  */
    bool verbose = false;
};

typedef enum {
    SOLVED,
    MAX_ITER_EXCEEDED,
    UNSOLVED,
    UNINITIALIZED
} status_t;

template <typename Scalar>
struct qp_solver_info_t {
    status_t status = UNINITIALIZED; /**< Solver status */
    int iter = 0;               /**< Number of iterations */
    int rho_updates = 0;        /**< Number of rho updates (factorizations) */
    Scalar rho_estimate = 0;    /**< Last rho estimate */
    Scalar res_prim = 0;        /**< Primal residual */
    Scalar res_dual = 0;        /**< Dual residual */
};

/**-----------------------------------------------------------------------------------*/
/** (Almost) Interface class for general QP solvers */
template<typename Derived, int N, int M, typename Scalar = double,
         template <typename, int, typename ...Args> class LinearSolver = Eigen::LDLT,
         int LinearSolver_UpLo = Eigen::Lower, typename ...Args>
class QPBase
{
  public:
    QPBase() = default;

    using scalar_t     = Scalar;
    using qp_var_t     = Eigen::Matrix<scalar_t, N, 1>;
    using qp_constraint_t = Eigen::Matrix<scalar_t, M, N>;
    using qp_dual_t    = Eigen::Matrix<scalar_t, M, 1>;
    using qp_kkt_vec_t = Eigen::Matrix<scalar_t, N + M, 1>;
    using kkt_mat_t    = Eigen::Matrix<scalar_t, N + M, N + M>;
    using kkt_vec_t    = Eigen::Matrix<scalar_t, N + M, 1>;
    using qp_hessian_t = Eigen::Matrix<scalar_t, N, N>;

    using settings_t = qp_sover_settings_t<scalar_t>;
    using info_t = qp_solver_info_t<scalar_t>;
    using linear_solver_t = LinearSolver<kkt_mat_t, LinearSolver_UpLo, Args...>;

    static constexpr scalar_t LOOSE_BOUNDS_THRESH = 1e+10;
    static constexpr scalar_t EQ_TOL = 1e-4;
    static constexpr scalar_t DIV_BY_ZERO_REGUL = std::numeric_limits<scalar_t>::epsilon();

    // Solver state variables
    qp_var_t  m_x;
    qp_dual_t m_y;

    using constraint_type = enum {
        INEQUALITY_CONSTRAINT,
        EQUALITY_CONSTRAINT,
        LOOSE_BOUNDS
    };

    constraint_type constr_type[M]; /**< constraint type classification */

    settings_t m_settings;
    info_t m_info;

    // enforce 16 byte alignment https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** getters  / setters */
    EIGEN_STRONG_INLINE const qp_var_t& primal_solution() const noexcept { return m_x; }
    EIGEN_STRONG_INLINE qp_var_t& primal_solution() noexcept { return m_x; }

    EIGEN_STRONG_INLINE const qp_dual_t& dual_solution() const { return m_y; }
    EIGEN_STRONG_INLINE qp_dual_t& dual_solution() noexcept { return m_y; }

    EIGEN_STRONG_INLINE const settings_t& settings() const noexcept { return m_settings; }
    EIGEN_STRONG_INLINE settings_t& settings() noexcept { return m_settings; }

    EIGEN_STRONG_INLINE const info_t& info() const noexcept { return m_info; }
    EIGEN_STRONG_INLINE info_t& info() noexcept { return m_info; }

    /** general functions*/
    /** solve */
    status_t solve(const Eigen::Ref<const qp_hessian_t>&H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>&A,
                   const Eigen::Ref<const qp_dual_t>& Alb, const Eigen::Ref<const qp_dual_t>& Aub) noexcept
    {
        return static_cast<Derived*>(this)->solve_impl(H, h, A, Alb, Aub);
    }
    /** solve using initial guess */
    status_t solve(const Eigen::Ref<const qp_hessian_t>&H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>&A,
                   const Eigen::Ref<const qp_dual_t>& Alb, const Eigen::Ref<const qp_dual_t>& Aub,
                   const Eigen::Ref<const qp_var_t>& x_guess, const Eigen::Ref<const qp_dual_t>& y_guess) noexcept
    {
        return static_cast<Derived*>(this)->solve_impl(H, h, A, Alb, Aub, x_guess, y_guess);
    }

    /** parse constraints bounds */
    EIGEN_STRONG_INLINE void parse_constraints_bounds(const Eigen::Ref<const qp_dual_t>& Alb, const Eigen::Ref<const qp_dual_t>& Aub) noexcept
    {
        eigen_assert((Alb.array() < Aub.array()).any());

        for (int i = 0; i < qp_dual_t::RowsAtCompileTime; i++)
        {
            if (Alb(i) < -LOOSE_BOUNDS_THRESH and Aub[i] > LOOSE_BOUNDS_THRESH)
                constr_type[i] = LOOSE_BOUNDS;
            else if (Aub[i] - Alb[i] < EQ_TOL)
                constr_type[i] = EQUALITY_CONSTRAINT;
            else
                constr_type[i] = INEQUALITY_CONSTRAINT;
        }

    }

    /** standard function to estimate primal residual */
    EIGEN_STRONG_INLINE scalar_t primal_residual(const Eigen::Ref<const qp_constraint_t>& A, const Eigen::Ref<const qp_var_t>& x,
                                                 const Eigen::Ref<const qp_dual_t>& b) const noexcept
    {
        return (A * x - b).template lpNorm<Eigen::Infinity>();
    }

    /** dual residual estimation */
    EIGEN_STRONG_INLINE scalar_t dual_residual(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h,
                                               const Eigen::Ref<const qp_constraint_t>& A, const Eigen::Ref<const qp_var_t>& x,
                                               const Eigen::Ref<const qp_dual_t>& y) const noexcept
    {
        return (H * x + h + A.transpose() * y).template lpNorm<Eigen::Infinity>();
    }

};





#endif // QP_BASE_HPP
