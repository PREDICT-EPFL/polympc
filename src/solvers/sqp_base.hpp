#ifndef SQP_BASE_HPP
#define SQP_BASE_HPP

#include <memory>
#include "Eigen/Core"
#include "bfgs.hpp"
#include "admm.hpp"
#include "box_admm.hpp"
#include "qp_base.hpp"
#include "utils.hpp"
#include <iostream>

template <typename Scalar>
struct sqp_settings_t {
    Scalar tau = 0.5;       /**< line search iteration decrease, 0 < tau < 1 */
    Scalar eta = 0.25;      /**< line search parameter, 0 < eta < 1 */
    Scalar rho = 0.5;       /**< line search parameter, 0 < rho < 1 */
    Scalar eps_prim = 1e-3; /**< primal step termination threshold, eps_prim > 0 */
    Scalar eps_dual = 1e-3; /**< dual step termination threshold, eps_dual > 0 */
    int max_iter = 100;
    int line_search_max_iter = 100;
    void (*iteration_callback)(void *solver) = nullptr;

    bool validate()
    {
        bool valid;
        valid = 0.0 < tau && tau < 1.0 &&
                0.0 < eta && eta < 1.0 &&
                0.0 < rho && rho < 1.0 &&
                eps_prim > 0.0 &&
                eps_dual > 0.0 &&
                max_iter > 0 &&
                line_search_max_iter > 0;
        return valid;
    }
};

struct sqp_status_t {
    enum {
        SOLVED,
        MAX_ITER_EXCEEDED,
        INVALID_SETTINGS
    } value;
};

struct sqp_info_t {
    int iter;
    int qp_solver_iter;
    sqp_status_t status;
};


template<typename Derived, typename Problem, typename QPSolver> class SQPBase;

template<typename Derived, typename Problem, typename QPSolver = boxADMM<Problem::VAR_SIZE, Problem::NUM_EQ + Problem::NUM_INEQ, typename Problem::scalar_t>>
class SQPBase
{
public:
    SQPBase()
    {
        /** set default constraints */
        /** @badcode: change to Constant */
        m_lbx = nlp_variable_t::Constant(static_cast<scalar_t>(-INF));
        m_ubx = nlp_variable_t::Constant(static_cast<scalar_t>( INF));
        /** @bug: fix initialisation */
        m_x.setOnes();
        m_lam.setZero();

        m_qp_solver.settings().warm_start = false;
        m_qp_solver.settings().check_termination = 10;
        m_qp_solver.settings().eps_abs = 1e-4;
        m_qp_solver.settings().eps_rel = 1e-4;
        m_qp_solver.settings().max_iter = 100;
        m_qp_solver.settings().adaptive_rho = true;
        m_qp_solver.settings().adaptive_rho_interval = 50;
        m_qp_solver.settings().alpha = 1.0; //1.6

        //std::cout << "QP size: " << sizeof (m_qp_solver) << "\n";
        if(Problem::is_sparse)
        {
            m_H.resize(VAR_SIZE, VAR_SIZE);
            m_A.resize(NUM_EQ + NUM_INEQ, VAR_SIZE);
        }

    }

  enum
  {
      VAR_SIZE   = Problem::VAR_SIZE,
      NUM_EQ     = Problem::NUM_EQ,
      NUM_INEQ   = Problem::NUM_INEQ,
      NUM_CONSTR = Problem::DUAL_SIZE
  };

  /** get main function from the derived class */
  using nlp_variable_t    = typename Problem::nlp_variable_t;
  using nlp_constraints_t = typename Problem::nlp_constraints_t;
  using nlp_eq_jacobian_t = typename Problem::nlp_eq_jacobian_t;
  using nlp_hessian_t     = typename Problem::nlp_hessian_t;
  using nlp_cost_t        = typename Problem::scalar_t;
  using nlp_dual_t        = typename Problem::nlp_dual_t;
  using scalar_t          = typename Problem::scalar_t;
  using parameter_t       = typename Problem::static_parameter_t;

  using qp_solver_t = QPSolver;
  using nlp_settings_t = sqp_settings_t<scalar_t>;
  using nlp_info_t     = sqp_info_t;

  /** instantiate the problem */
  Problem problem;

  nlp_hessian_t     m_H;  // Hessian of Lagrangian
  nlp_variable_t    m_h;  // Gradient of the cost function
  nlp_variable_t    m_x;  // variable primal
  nlp_dual_t        m_lam, m_lam_k; // dual variable
  nlp_eq_jacobian_t m_A;   // equality constraints Jacobian
  nlp_constraints_t m_b;   // equality constraints evaluated
  parameter_t       m_p = parameter_t::Zero(); // problem parameters
  scalar_t          m_cost;
  nlp_settings_t    m_settings;
  sqp_info_t        m_info;
  nlp_variable_t    m_lbx, m_ubx;

  /** QP solver */
  qp_solver_t m_qp_solver;

  /** temporary storage for Lagrange gradient */
  nlp_variable_t m_lag_gradient;
  nlp_variable_t m_step_prev;
  scalar_t m_primal_norm;
  scalar_t m_dual_norm;


  static constexpr scalar_t EPSILON = std::numeric_limits<scalar_t>::epsilon();
  static constexpr scalar_t INF     = std::numeric_limits<scalar_t>::infinity();

  /** getters / setters */
  EIGEN_STRONG_INLINE const Problem& get_problem() const noexcept { return this->problem; }
  EIGEN_STRONG_INLINE Problem& get_problem() noexcept { return this->problem; }

  EIGEN_STRONG_INLINE const nlp_variable_t& primal_solution() const noexcept { return m_x; }
  EIGEN_STRONG_INLINE nlp_variable_t& primal_solution() noexcept { return m_x; }

  EIGEN_STRONG_INLINE const nlp_dual_t& dual_solution() const noexcept { return m_lam; }
  EIGEN_STRONG_INLINE nlp_dual_t& dual_solution() noexcept { return m_lam; }

  EIGEN_STRONG_INLINE const nlp_settings_t& settings() const noexcept { return m_settings; }
  EIGEN_STRONG_INLINE nlp_settings_t& settings() noexcept { return m_settings; }

  EIGEN_STRONG_INLINE const sqp_info_t& info() const noexcept { return m_info; }
  EIGEN_STRONG_INLINE sqp_info_t& info() noexcept { return m_info; }

  EIGEN_STRONG_INLINE const nlp_variable_t& lower_bound_x() const noexcept { return m_lbx; }
  EIGEN_STRONG_INLINE nlp_variable_t& lower_bound_x() noexcept { return m_lbx; }

  EIGEN_STRONG_INLINE const nlp_variable_t& upper_bound_x() const noexcept { return m_ubx; }
  EIGEN_STRONG_INLINE nlp_variable_t& upper_bound_x() noexcept { return m_ubx; }

  EIGEN_STRONG_INLINE const parameter_t& parameters() const noexcept { return m_p; }
  EIGEN_STRONG_INLINE parameter_t& parameters() noexcept { return m_p; }

  EIGEN_STRONG_INLINE const typename qp_solver_t::settings_t& qp_settings() const noexcept { return m_qp_solver.m_settings; }
  EIGEN_STRONG_INLINE typename qp_solver_t::settings_t& qp_settings() noexcept { return m_qp_solver.m_settings; }


  /** step size selection: line search / filter / trust resion */
  scalar_t step_size_selection(const Eigen::Ref<const nlp_variable_t>& p) noexcept
  {
      return static_cast<Derived*>(this)->step_size_selection_impl(p);
  }
  scalar_t step_size_selection(const Eigen::Ref<const nlp_variable_t>& p) const noexcept
  {
      return static_cast<const Derived*>(this)->step_size_selection_impl(p);
  }

  /** evaluate constraints violation */
  EIGEN_STRONG_INLINE scalar_t constraints_violation(const Eigen::Ref<const nlp_variable_t>& x) const noexcept
  {
      return static_cast<const Derived*>(this)->constraints_violation_impl(x);
  }

  /** maximum constraints violation */
  EIGEN_STRONG_INLINE scalar_t max_constraints_violation(const Eigen::Ref<const nlp_variable_t>& x) const noexcept
  {
      return static_cast<const Derived*>(this)->max_constraints_violation_impl(x);
  }

  /** linearisation -> forming local QP */
  template<int T = Problem::MATRIXFMT>
  EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
  linearisation(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                 const Eigen::Ref<const nlp_dual_t>& lam,
                                 Eigen::Ref<nlp_variable_t> cost_grad, nlp_hessian_t& lag_hessian,
                                 nlp_eq_jacobian_t& Ae, Eigen::Ref<nlp_constraints_t> be) noexcept
  {
      static_cast<Derived*>(this)->linearisation_sparse_impl(x, p, lam, cost_grad, lag_hessian, Ae, be);
  }

  template<int T = Problem::MATRIXFMT>
  EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
  linearisation(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                 const Eigen::Ref<const nlp_dual_t>& lam,
                                 Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                 Eigen::Ref<nlp_eq_jacobian_t> Ae, Eigen::Ref<nlp_constraints_t> be) noexcept
  {
      static_cast<Derived*>(this)->linearisation_dense_impl(x, p, lam, cost_grad, lag_hessian, Ae, be);
  }

  /** update linearisation: option for faster linearisation update*/
  EIGEN_STRONG_INLINE void update_linearisation(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                            const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                            Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                            typename std::conditional<Problem::MATRIXFMT == DENSE, Eigen::Ref<nlp_eq_jacobian_t>, nlp_eq_jacobian_t&>::type Ae,
                            Eigen::Ref<nlp_constraints_t> be) noexcept
  {
      static_cast<Derived*>(this)->update_linearisation_impl(x, p, x_step, lam, cost_grad, lag_hessian, Ae, be);
  }

  /** Hessian update: room for creativity: L-BFGS, BFGS (default), sparse BFGS, SR1 */
  EIGEN_STRONG_INLINE void hessian_update(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t>& x_step,
                                          const Eigen::Ref<const nlp_variable_t>& grad_step) noexcept
  {
      static_cast<Derived*>(this)->hessian_update_impl(hessian, x_step, grad_step);
  }

  /** termination criteria */
  EIGEN_STRONG_INLINE bool termination_criteria(const Eigen::Ref<const nlp_variable_t>& x) const noexcept
  {
      return static_cast<const Derived*>(this)->termination_criteria_impl(x);
  }

  /** Hessian regularisation if necessary: default behavior: do nothing*/
  template<int T = Problem::MATRIXFMT>
  EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
  hessian_regularisation(Eigen::Ref<nlp_hessian_t> lag_hessian)
  {
      static_cast<Derived*>(this)->hessian_regularisation_dense_impl(lag_hessian);
  }

  template<int T = Problem::MATRIXFMT>
  EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
  hessian_regularisation(nlp_hessian_t& lag_hessian)
  {
      static_cast<Derived*>(this)->hessian_regularisation_sparse_impl(lag_hessian);
  }

  /**
   * default implementations
   */

  /** default algorithm: l1-norm line search */
  scalar_t step_size_selection_impl(const Eigen::Ref<const nlp_variable_t>& p) noexcept;

  /** default constraints violation */
  EIGEN_STRONG_INLINE scalar_t constraints_violation_impl(const Eigen::Ref<const nlp_variable_t>& x) const noexcept;

  /** default max constraint violation */
  EIGEN_STRONG_INLINE scalar_t max_constraints_violation_impl(const Eigen::Ref<const nlp_variable_t>& x) const noexcept;

  /** default regularisation: do nothing */
  EIGEN_STRONG_INLINE void hessian_regularisation_dense_impl(Eigen::Ref<nlp_hessian_t> lag_hessian) noexcept {}
  EIGEN_STRONG_INLINE void hessian_regularisation_sparse_impl(Eigen::Ref<nlp_hessian_t> lag_hessian) noexcept {}

  /** default linearisation routine: exact linearisation with AD */
  EIGEN_STRONG_INLINE void
  linearisation_dense_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                              const Eigen::Ref<const nlp_dual_t>& lam,
                                              Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                              Eigen::Ref<nlp_eq_jacobian_t> Ae, Eigen::Ref<nlp_constraints_t> be) noexcept
  {
      scalar_t _lag(0.0);
      problem.lagrangian_gradient_hessian(x,p,lam, _lag, m_lag_gradient, lag_hessian, cost_grad, be, Ae);
  }

  // sparse version
  EIGEN_STRONG_INLINE void
  linearisation_sparse_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                            const Eigen::Ref<const nlp_dual_t>& lam,
                            Eigen::Ref<nlp_variable_t> cost_grad, nlp_hessian_t& lag_hessian,
                            nlp_eq_jacobian_t& Ae, Eigen::Ref<nlp_constraints_t> be) noexcept
  {
      scalar_t _lag;
      problem.lagrangian_gradient_hessian(x,p,lam, _lag, m_lag_gradient, lag_hessian, cost_grad, be, Ae);
  }

  /** default linearisation update uses damped BFGS algorithm */
  EIGEN_STRONG_INLINE void update_linearisation_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                 const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                                 Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                 typename std::conditional<Problem::MATRIXFMT == DENSE, Eigen::Ref<nlp_eq_jacobian_t>, nlp_eq_jacobian_t&>::type Ae,
                                 Eigen::Ref<nlp_constraints_t> be) noexcept;

  /** default Hessain update -> dense damped BFGS */
  EIGEN_STRONG_INLINE void hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t>& x_step,
                                               const Eigen::Ref<const nlp_variable_t>& grad_step) noexcept
  {
        BFGS_update(hessian, x_step, grad_step);
  }

  EIGEN_STRONG_INLINE bool termination_criteria_impl(const Eigen::Ref<const nlp_variable_t>& x) const noexcept;

  bool _is_posdef(const Eigen::Ref<const nlp_hessian_t>& H) const noexcept
  {
      Eigen::EigenSolver<nlp_hessian_t> eigensolver(H);
      for (int i = 0; i < eigensolver.eigenvalues().rows(); i++) {
          double v = eigensolver.eigenvalues()(i).real();
          if (v <= 0) {
              return false;
          }
      }
      return true;
  }

  /** prepare and solve the qp */
  void solve_qp(Eigen::Ref<nlp_variable_t> prim_step, Eigen::Ref<nlp_dual_t> dual_step) noexcept;

  /** finally solve the NLP */
  void solve() noexcept;
  void solve(const Eigen::Ref<const nlp_variable_t>& x_guess, const Eigen::Ref<const nlp_dual_t>& lam_guess) noexcept
  {
      m_x = x_guess;
      m_lam = lam_guess;
      this->solve();
  }

};

template<typename Derived, typename Problem, typename QPSolver>
typename SQPBase<Derived, Problem, QPSolver>::scalar_t
SQPBase<Derived, Problem, QPSolver>::step_size_selection_impl(const Eigen::Ref<const nlp_variable_t>& p) noexcept
{
    scalar_t mu, phi_l1, Dp_phi_l1;
    nlp_variable_t cost_gradient = m_h;
    const scalar_t tau = m_settings.tau; // line search step decrease, 0 < tau < settings.tau

    scalar_t constr_l1 = constraints_violation(m_x);

    // TODO: get mu from merit function model using hessian of Lagrangian
    mu = abs(cost_gradient.dot(p)) / ((1 - m_settings.rho) * constr_l1);

    scalar_t cost_1;
    problem.cost(m_x, m_p, cost_1);

    //std::cout << "l1: " << constr_l1 << " cost: " << cost_1 << "\n";

    phi_l1 = cost_1 + mu * constr_l1;
    Dp_phi_l1 = cost_gradient.dot(p) - mu * constr_l1;

    scalar_t alpha = scalar_t(1.0);
    scalar_t cost_step;
    nlp_variable_t x_step;
    for (int i = 1; i < m_settings.line_search_max_iter; i++)
    {
        x_step.noalias() = alpha * p;
        x_step += m_x;
        problem.cost(x_step, m_p, cost_step);

        //std::cout << "l1: " << constraints_violation(x_step) << " cost: " << cost_step << "\n";

        scalar_t phi_l1_step = cost_step + mu * constraints_violation(x_step);

        //std::cout << "phi before: " << phi_l1 << " after: " << phi_l1_step <<  " required diff: " << alpha * m_settings.eta * Dp_phi_l1 << "\n";

        if (phi_l1_step <= (phi_l1 + alpha * m_settings.eta * Dp_phi_l1))
        {
            // accept step
            return alpha;
        } else {
            alpha = tau * alpha;
        }
    }

    return alpha;
}

template<typename Derived, typename Problem, typename QPSolver>
typename SQPBase<Derived, Problem, QPSolver>::scalar_t
SQPBase<Derived, Problem, QPSolver>::constraints_violation_impl(const Eigen::Ref<const nlp_variable_t>& x) const noexcept
{
    scalar_t cl1 = EPSILON;
    nlp_constraints_t c_eq;
    //constr_ineq_t c_ineq;

    problem.equalities(x, m_p, c_eq);

    // c_eq = 0
    cl1 += c_eq.template lpNorm<1>();

    /**
    // c_ineq <= 0
    cl1 += c_ineq.cwiseMax(0.0).sum();
    */

    // l <= x <= u
    cl1 += (m_lbx - x).cwiseMax(0.0).sum();
    cl1 += (x - m_ubx).cwiseMax(0.0).sum();

    return cl1;
}

template<typename Derived, typename Problem, typename QPSolver>
typename SQPBase<Derived, Problem, QPSolver>::scalar_t
SQPBase<Derived, Problem, QPSolver>::max_constraints_violation_impl(const Eigen::Ref<const nlp_variable_t>& x) const noexcept
{
    scalar_t c = scalar_t(0);
    nlp_constraints_t c_eq;
    //constr_ineq_t c_ineq;

    problem.equalities(x, m_p, c_eq);

    // c_eq = 0
    if (NUM_EQ > 0) {
        c = c_eq.template lpNorm<Eigen::Infinity>();
    }

    // c_ineq <= 0
    /**
    if (NUM_INEQ > 0) {
        c = fmax(c, c_ineq.maxCoeff());;
    }*/

    // l <= x <= u
    c = fmax(c, (m_lbx - x).maxCoeff());
    c = fmax(c, (x - m_ubx).maxCoeff());

    return c;
}

/**
template<typename Derived, typename Problem, typename QPSolver>
void SQPBase<Derived, Problem, QPSolver>::linearisation_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                                   const Eigen::Ref<const nlp_dual_t>& lam,
                                                   Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                   Eigen::Ref<nlp_eq_jacobian_t> Ae, Eigen::Ref<nlp_constraints_t> be) noexcept
{
    scalar_t _lag;
    problem.lagrangian_gradient_hessian(x,p,lam, _lag, m_lag_gradient, lag_hessian, cost_grad, be, Ae);
    //std::cout << "optimality: " << m_lag_gradient.template lpNorm<Eigen::Infinity>() << "\n";
}
*/

template<typename Derived, typename Problem, typename QPSolver>
void SQPBase<Derived, Problem, QPSolver>::update_linearisation_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                                          const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                                                          Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                          typename std::conditional<Problem::MATRIXFMT == DENSE,
                                                          Eigen::Ref<nlp_eq_jacobian_t>, nlp_eq_jacobian_t&>::type Ae,
                                                          Eigen::Ref<nlp_constraints_t> be) noexcept
{
    scalar_t _lag;
    nlp_variable_t _lag_grad;
    problem.lagrangian_gradient(x, p, lam, _lag, _lag_grad, cost_grad, be, Ae);

    /** @badcode: redo with gradient step:  BFGS update */
    //std::cout << "optimality: " << _lag_grad.template lpNorm<Eigen::Infinity>() << "\n";
    hessian_update(lag_hessian, x_step, (_lag_grad - m_lag_gradient));
    m_lag_gradient = _lag_grad;
}

template<typename Derived, typename Problem, typename QPSolver>
bool SQPBase<Derived, Problem, QPSolver>::termination_criteria_impl(const Eigen::Ref<const nlp_variable_t>& x) const noexcept
{
    //std::cout << "residuals: " << m_primal_norm << " " << m_dual_norm << " " << max_constraints_violation(x) << "\n";
    return (m_primal_norm <= m_settings.eps_prim) && (m_dual_norm <= m_settings.eps_dual) &&
            (max_constraints_violation(x) <= m_settings.eps_prim) ? true : false;
}


template<typename Derived, typename Problem, typename QPSolver>
void SQPBase<Derived, Problem, QPSolver>::solve_qp(Eigen::Ref<nlp_variable_t> prim_step, Eigen::Ref<nlp_dual_t> dual_step) noexcept
{
    /**
    enum {
        EQ_IDX = 0,
        INEQ_IDX = NUM_EQ,
        BOX_IDX = NUM_INEQ + NUM_EQ,
    };
    const scalar_t UNBOUNDED = std::numeric_limits<scalar_t>::max();

    // Inequality constraints
    // from          A.x + b <= 0
    // to    -INF <= A.x     <= -b

    _qp.u.template segment<NUM_INEQ>(INEQ_IDX) = -b_ineq;
    _qp.l.template segment<NUM_INEQ>(INEQ_IDX).setConstant(-UNBOUNDED);
    _qp.A.template block<NUM_INEQ, VAR_SIZE>(INEQ_IDX, 0) = A_ineq;
    */

    // solve the QP
    //typename qp_solver_t::status_t qp_status;
    status_t qp_status;

    //std::cout << " \n QP: \n";
    //std::cout << "H: \n" << m_H << "\n";
    //std::cout << "h: " << m_h.transpose() << "\n";
    //std::cout << "A: " << m_A << "\n";
    //std::cout << "b: " << -m_b.transpose() << "\n";
    //std::cout << "lx: " << (m_lbx - m_x).transpose() << " | ux: " << (m_ubx - m_x).transpose() << "\n";


    /** @badcode: m_x -> x step; m_lam -> lam step */
    qp_status = m_qp_solver.solve(m_H, m_h, m_A, -m_b, -m_b, m_lbx - m_x, m_ubx - m_x);
    //std::cout << "QP status: " << qp_status << " QP iter: " << m_qp_solver.info().iter << "\n";

    eigen_assert(qp_status == status_t::SOLVED);

    m_info.qp_solver_iter += m_qp_solver.info().iter;

    prim_step = m_qp_solver.primal_solution();
    dual_step = m_qp_solver.dual_solution();

    //std::cout << "dx: " << prim_step.transpose() << " | dy: " << dual_step.transpose() << "\n";
}

/** solve method */
template<typename Derived, typename Problem, typename QPSolver>
void SQPBase<Derived, Problem, QPSolver>::solve() noexcept
{
    m_info.status.value = sqp_status_t::MAX_ITER_EXCEEDED;

    nlp_variable_t p;   // search direction
    nlp_dual_t p_lambda; // dual search direction
    scalar_t alpha;    // step size

    p_lambda.setZero();

    // initialize
    m_info.qp_solver_iter = 0;
    m_info.iter = 1;

    //std::cout << "x:" << m_x.transpose() << "\n";

    /** solve once with exact linearisation */
    //std::cout << "Linerise at: x:" << m_x.transpose() << " | y:" << m_lam.transpose() << "\n";

    linearisation(m_x, m_p, m_lam, m_h, m_H, m_A, m_b);
    /** place for heuristics: regularisation and preconditioning */
    hessian_regularisation(m_H);

    solve_qp(p, p_lambda);
    // in the case the sparse solver is used: reuse sparsity pattern for further solves
    m_qp_solver.settings().reuse_pattern = true;
    m_qp_solver.settings().warm_start    = true; // for other solvers

    m_lam_k = p_lambda;

    //std::cout << "x_step: " << p.transpose() << "\n";

    p_lambda -= m_lam;
    alpha = step_size_selection(p);

    //alpha = scalar_t(1.0);

    //std::cout << "alpha: " << alpha << "\n";

    // take step
    m_x.noalias()   += alpha * p;
    m_lam.noalias() += alpha * p_lambda;

    // update step info
    m_step_prev.noalias() = alpha * p;
    m_primal_norm = alpha * p.template lpNorm<Eigen::Infinity>();
    m_dual_norm   = alpha * p_lambda.template lpNorm<Eigen::Infinity>();

    //std::cout << "x:" << m_x.transpose() << "\n";


    if (termination_criteria(m_x)) {
        m_info.status.value = sqp_status_t::SOLVED;
        return;
    }

    //std::cout << m_x.transpose() << "\n";

    while (m_info.iter < m_settings.max_iter)
    {
        m_info.iter++;
        /** linearise and solve qp here*/
        update_linearisation(m_x, m_p, m_step_prev, m_lam, m_h, m_H, m_A, m_b);

        //linearisation(m_x, m_p, m_lam, m_h, m_H, m_A, m_b);
        /** place for heuristics: regularisation and preconditioning */
        //hessian_regularisation(m_H);

        solve_qp(p, p_lambda);

        /** trial */
        m_lam_k = p_lambda;

        p_lambda -= m_lam;
        alpha = step_size_selection(p);

        // take step
        m_x.noalias()   += alpha * p;
        m_lam.noalias() += alpha * p_lambda; // was "-"

        // update step info
        m_step_prev.noalias() = alpha * p;
        m_primal_norm = alpha * p.template lpNorm<Eigen::Infinity>();
        m_dual_norm   = alpha * p_lambda.template lpNorm<Eigen::Infinity>();

        if (m_settings.iteration_callback != nullptr)
            m_settings.iteration_callback(this);


        if (termination_criteria(m_x)) {
            m_info.status.value = sqp_status_t::SOLVED;
            break;
        }
    }

    //if (m_info.iter > m_settings.max_iter)
    //    m_info.status.value = sqp_status_t::MAX_ITER_EXCEEDED;
}


#endif // SQP_BASE_HPP
