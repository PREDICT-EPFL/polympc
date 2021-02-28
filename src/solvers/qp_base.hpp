#ifndef QP_BASE_HPP
#define QP_BASE_HPP

#include "utils/helpers.hpp"
#include <iostream>

template <typename Scalar>
struct qp_solver_settings_t {

    /** Common settings */
    Scalar eps_rel = 1e-3;      /**< Relative tolerance for termination, 0 < eps_rel */
    Scalar eps_abs = 1e-3;      /**< Absolute tolerance for termination, 0 < eps_abs */
    int max_iter = 1000;        /**< Maximal number of iteration, 0 < max_iter */
    bool warm_start    = false; /**< Warm start solver, reuses previous x,z,y */
    bool reuse_pattern = false; /**< Assume that problem size and sparsity pattern have not changed since last 'solve call' */
    bool verbose = false;

    /** ADMM related */
    Scalar rho = 1e-1;          /**< ADMM rho step, 0 < rho */
    Scalar sigma = 1e-6;        /**< ADMM sigma step, 0 < sigma, (small) */
    Scalar alpha = 1.0;         /**< ADMM overrelaxation parameter, 0 < alpha < 2,
                                     values in [1.5, 1.8] give good results (empirically) */    
    int check_termination = 25; /**< Check termination after every Nth iteration, 0 (disabled) or 0 < check_termination */
    bool adaptive_rho = false;  /**< Adapt rho to optimal estimate */
    Scalar adaptive_rho_tolerance = 5;  /**< Minimal for rho update factor, 1 < adaptive_rho_tolerance */
    int adaptive_rho_interval = 25; /**< change rho every Nth iteration, 0 < adaptive_rho_interval,
                                         set equal to check_termination to save computation  */

    /** OSQP specific */
    Scalar delta      = 1e-6;           /** Polishing regularization parameter */
    bool   polish     = false;          /** Perform polishing */
    int    scaling    = 10;             /** Number of scaling iterations */
    Scalar time_limit = 0;              /** Run time limit in seconds */
    int polish_refine_iter = 3;         /** Refinement iterations in polish */
    int osqp_linear_solver = 0;         /** Linear systems solver type: 0 = LDLT, 1 = Pardiso */
    bool scaled_termination = false;    /** Scaled termination conditions */
    Scalar adaptive_rho_fraction = 0.4; /** Adaptive rho interval as fraction of setup time (auto mode) */
    Scalar eps_dual_inf = 1e-4;         /** Primal infeasibility tolerance */
    Scalar eps_prim_inf = 1e-4;         /** Dual infeasibility tolerance */

    /** Goldfarb-Idnani Active-Set solver (QPMAD) */
    int hessian_type = 1;       /** 0: UNDEFINED | 1: LOWER_TRIANGULAR | 2: CHOLESKY_FACTOR  (default = 1) */
};

typedef enum {
    SOLVED,
    MAX_ITER_EXCEEDED,
    UNSOLVED,
    UNINITIALIZED,
    INFEASIBLE,
    INCONSISTENT
} status_t;

template <typename Scalar>
struct qp_solver_info_t {
    status_t status = UNINITIALIZED; /**< Solver status */
    int iter = 0;               /**< Number of iterations */
    int rho_updates = 0;        /**< Number of rho updates (factorizations) */
    Scalar rho_estimate = 0;    /**< Last rho estimate */
    Scalar res_prim = 1;        /**< Primal residual */
    Scalar res_dual = 1;        /**< Dual residual */
};

/**-----------------------------------------------------------------------------------*/
/** (Almost) Interface class for generic QP solvers
 * N - state dimension
 * M - numer of generic constraints
*/

template<typename Derived, int N, int M, typename Scalar = double, int MatrixType = DENSE,
         template <typename, int, typename ...Args> class LinearSolver = linear_solver_traits<DENSE>::default_solver,
         int LinearSolver_UpLo = Eigen::Lower, typename ...Args>
class QPBase
{
  public:
    QPBase() = default;

    using scalar_t     = Scalar;
    using qp_var_t     = typename dense_matrix_type_selector<scalar_t, N, 1>::type;
    using qp_dual_t    = typename dense_matrix_type_selector<scalar_t, N + M, 1>::type;
    using qp_dual_a_t  = typename dense_matrix_type_selector<scalar_t, M, 1>::type;
    using qp_kkt_vec_t = typename dense_matrix_type_selector<scalar_t, N + M, 1>::type;
    using kkt_vec_t    = typename dense_matrix_type_selector<scalar_t, N + M, 1>::type;
    using qp_constraint_t = typename std::conditional<MatrixType == SPARSE, Eigen::SparseMatrix<scalar_t>,
                            typename dense_matrix_type_selector<scalar_t, M, N>::type>::type;

    using qp_hessian_t    = typename std::conditional<MatrixType == SPARSE, Eigen::SparseMatrix<scalar_t>,
                            typename dense_matrix_type_selector<scalar_t, N, N>::type>::type;

    using kkt_mat_t       = typename std::conditional<MatrixType == SPARSE, Eigen::SparseMatrix<scalar_t>,
                            typename dense_matrix_type_selector<scalar_t, N + M, N + M>::type>::type;

    using settings_t = qp_solver_settings_t<scalar_t>;
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

    settings_t m_settings;
    info_t m_info;

    constraint_type box_constr_type[N]; /** box constraints parsing */
    constraint_type constr_type[M]; /**< constraint type classification */

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

    /** solve with generic and box constraints*/
    status_t solve(const Eigen::Ref<const qp_hessian_t>&H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
                   const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub,
                   const Eigen::Ref<const qp_var_t>& xlb, const Eigen::Ref<const qp_var_t>& xub) noexcept
    {
        return static_cast<Derived*>(this)->solve_impl(H, h, A, Alb, Aub, xlb, xub);
    }

    /** solve with an intial guess */
    status_t solve(const Eigen::Ref<const qp_hessian_t>&H, const Eigen::Ref<const qp_var_t>& h, const Eigen::Ref<const qp_constraint_t>& A,
                   const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub,
                   const Eigen::Ref<const qp_var_t>& xlb, const Eigen::Ref<const qp_var_t>& xub,
                   const Eigen::Ref<const qp_var_t>& x_guess, const Eigen::Ref<const qp_dual_t>& y_guess) noexcept
    {
        return static_cast<Derived*>(this)->solve_impl(H, h, A, Alb, Aub, xlb, xub, x_guess, y_guess);
    }


    /** parse constraints bounds */
    EIGEN_STRONG_INLINE void parse_constraints_bounds(const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub) noexcept
    {
        eigen_assert((Alb.array() <= Aub.array()).any());

        for (int i = 0; i < qp_dual_a_t::RowsAtCompileTime; i++)
        {
            if (Alb(i) < -LOOSE_BOUNDS_THRESH && Aub[i] > LOOSE_BOUNDS_THRESH)
                constr_type[i] = LOOSE_BOUNDS;
            else if (Aub[i] - Alb[i] < EQ_TOL)
                constr_type[i] = EQUALITY_CONSTRAINT;
            else
                constr_type[i] = INEQUALITY_CONSTRAINT;
        }

    }

    EIGEN_STRONG_INLINE void parse_constraints_bounds(const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub,
                                                      const Eigen::Ref<const qp_var_t>& xlb, const Eigen::Ref<const qp_var_t>& xub) noexcept
    {
        eigen_assert((Alb.array() <= Aub.array()).any() xor (M <= 0)) ;
        eigen_assert((xlb.array() <= xub.array()).any());

        for (int i = 0; i < qp_dual_a_t::RowsAtCompileTime; i++)
        {
            if (Alb(i) < -LOOSE_BOUNDS_THRESH && Aub(i) > LOOSE_BOUNDS_THRESH)
                constr_type[i] = LOOSE_BOUNDS;
            else if (Aub[i] - Alb[i] < EQ_TOL)
                constr_type[i] = EQUALITY_CONSTRAINT;
            else
                constr_type[i] = INEQUALITY_CONSTRAINT;
        }

        /** parse box constraints*/
        for (int i = 0; i < qp_var_t::RowsAtCompileTime; i++)
        {
            if (xlb(i) < -LOOSE_BOUNDS_THRESH && xub(i) > LOOSE_BOUNDS_THRESH)
                box_constr_type[i] = LOOSE_BOUNDS;
            else if (xub[i] - xlb[i] < EQ_TOL)
                box_constr_type[i] = EQUALITY_CONSTRAINT;
            else
                box_constr_type[i] = INEQUALITY_CONSTRAINT;
        }

    }

    /** standard function to estimate primal residual */
    EIGEN_STRONG_INLINE scalar_t primal_residual(const Eigen::Ref<const qp_constraint_t>& A, const Eigen::Ref<const qp_var_t>& x,
                                                 const Eigen::Ref<const qp_dual_a_t>& b) const noexcept
    {
        return static_cast<const Derived*>(this)->primal_residual_impl(A, x, b);
    }

    /** dual residual estimation */
    EIGEN_STRONG_INLINE scalar_t dual_residual(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h,
                                               const Eigen::Ref<const qp_constraint_t>& A, const Eigen::Ref<const qp_var_t>& x,
                                               const Eigen::Ref<const qp_dual_t>& y) const noexcept
    {
        return static_cast<const Derived*>(this)->dual_residual_impl(H, h, A, x, y);
    }

    /** default implementation of the primal and dual residuals*/
    EIGEN_STRONG_INLINE scalar_t primal_residual_impl(const Eigen::Ref<const qp_constraint_t>& A, const Eigen::Ref<const qp_var_t>& x,
                                                 const Eigen::Ref<const qp_dual_a_t>& b) const noexcept
    {
        return (A * x - b).template lpNorm<Eigen::Infinity>();
    }

    /** dual residual estimation */
    EIGEN_STRONG_INLINE scalar_t dual_residual_impl(const Eigen::Ref<const qp_hessian_t>& H, const Eigen::Ref<const qp_var_t>& h,
                                               const Eigen::Ref<const qp_constraint_t>& A, const Eigen::Ref<const qp_var_t>& x,
                                               const Eigen::Ref<const qp_dual_t>& y) const noexcept
    {
        return (H * x + h + A.transpose() * y.template head<M>() + y.template tail<N>()).template lpNorm<Eigen::Infinity>();
    }

};





#endif // QP_BASE_HPP
