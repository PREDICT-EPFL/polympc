// This file is part of PolyMPC, a lightweight C++ template library
// for real-time nonlinear optimization and optimal control.
//
// Copyright (C) 2020 Listov Petr <petr.listov@epfl.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SQP_BASE_HPP
#define SQP_BASE_HPP

#include <memory>
#include <iostream>

#include "Eigen/Core"
#include "bfgs.hpp"
#include "admm.hpp"
#include "box_admm.hpp"
#include "qp_base.hpp"
#include "qp_preconditioners.hpp"
#include "utils.hpp"

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


template<typename Derived, typename Problem, typename QPSolver, typename Preconditioner> class SQPBase;

template<typename Derived, typename Problem,
         typename QPSolver = boxADMM<Problem::VAR_SIZE, Problem::NUM_EQ + Problem::NUM_INEQ, typename Problem::scalar_t>,
         typename Preconditioner = polympc::IdentityPreconditioner>
class SQPBase
{
public:
    SQPBase()
    {
        /** set default constraints */
        m_lbx = nlp_variable_t::Constant(static_cast<scalar_t>(-INF));
        m_ubx = nlp_variable_t::Constant(static_cast<scalar_t>( INF));
        m_lbg = nlp_ineq_constraints_t::Constant(static_cast<scalar_t>(-INF));
        m_ubg = nlp_ineq_constraints_t::Constant(static_cast<scalar_t>( INF));

        m_x.setZero();
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
  using nlp_eq_constraints_t   = typename Problem::nlp_eq_constraints_t;
  using nlp_ineq_constraints_t = typename Problem::nlp_ineq_constraints_t;
  using nlp_eq_jacobian_t = typename Problem::nlp_eq_jacobian_t;
  using nlp_jacobian_t    = typename Problem::nlp_jacobian_t;
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

  nlp_hessian_t     m_H;            // Hessian of Lagrangian
  nlp_variable_t    m_h;            // Gradient of the cost function
  nlp_variable_t    m_x;            // variable primal
  nlp_dual_t        m_lam, m_lam_k; // dual variable
  nlp_jacobian_t    m_A;            // equality constraints Jacobian
  nlp_constraints_t m_al, m_au;     // equality/inequality constraints evaluated
  parameter_t       m_p = parameter_t::Zero(); // problem parameters
  scalar_t          m_cost{0};
  nlp_settings_t    m_settings;
  sqp_info_t        m_info;
  nlp_variable_t    m_lbx, m_ubx;
  nlp_variable_t    m_lx, m_ux; // internal lower and upper bound for x
  nlp_ineq_constraints_t m_lbg, m_ubg;

  /** QP solver */
  qp_solver_t m_qp_solver;
  Preconditioner m_preconditioner;

  /** temporary storage for Lagrange gradient */
  nlp_variable_t m_lag_gradient;
  nlp_variable_t m_step_prev;
  scalar_t m_primal_norm;
  scalar_t m_dual_norm;
  scalar_t m_max_violation{0};


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

  EIGEN_STRONG_INLINE const nlp_ineq_constraints_t& lower_bound_g() const noexcept { return m_lbg; }
  EIGEN_STRONG_INLINE nlp_ineq_constraints_t& lower_bound_g() noexcept { return m_lbg; }

  EIGEN_STRONG_INLINE const nlp_ineq_constraints_t& upper_bound_g() const noexcept { return m_ubg; }
  EIGEN_STRONG_INLINE nlp_ineq_constraints_t& upper_bound_g() noexcept { return m_ubg; }

  EIGEN_STRONG_INLINE const parameter_t& parameters() const noexcept { return m_p; }
  EIGEN_STRONG_INLINE parameter_t& parameters() noexcept { return m_p; }

  EIGEN_STRONG_INLINE const typename qp_solver_t::settings_t& qp_settings() const noexcept { return m_qp_solver.m_settings; }
  EIGEN_STRONG_INLINE typename qp_solver_t::settings_t& qp_settings() noexcept { return m_qp_solver.m_settings; }

  EIGEN_STRONG_INLINE const scalar_t primal_norm() const noexcept {return m_primal_norm;}
  EIGEN_STRONG_INLINE const scalar_t dual_norm()   const noexcept {return m_dual_norm;}
  EIGEN_STRONG_INLINE const scalar_t constr_violation() const noexcept {return  m_max_violation;}
  EIGEN_STRONG_INLINE const scalar_t cost() const noexcept {return m_cost;}


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
                                 nlp_jacobian_t& A, Eigen::Ref<nlp_constraints_t> b) noexcept
  {
      static_cast<Derived*>(this)->linearisation_sparse_impl(x, p, lam, cost_grad, lag_hessian, A, b);
  }

  template<int T = Problem::MATRIXFMT>
  EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
  linearisation(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                 const Eigen::Ref<const nlp_dual_t>& lam,
                                 Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                 Eigen::Ref<nlp_jacobian_t> A, Eigen::Ref<nlp_constraints_t> b) noexcept
  {
      static_cast<Derived*>(this)->linearisation_dense_impl(x, p, lam, cost_grad, lag_hessian, A, b);
  }


  /** update linearisation: option for faster linearisation update*/
  template<int T = Problem::MATRIXFMT>
  EIGEN_STRONG_INLINE typename std::enable_if<T == DENSE>::type
  update_linearisation(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                       const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                       Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                       Eigen::Ref<nlp_jacobian_t> A, Eigen::Ref<nlp_constraints_t> b) noexcept
    {
        static_cast<Derived*>(this)->update_linearisation_dense_impl(x, p, x_step, lam, cost_grad, lag_hessian, A, b);
    }

  template<int T = Problem::MATRIXFMT>
  EIGEN_STRONG_INLINE typename std::enable_if<T == SPARSE>::type
  update_linearisation(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                       const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                       Eigen::Ref<nlp_variable_t> cost_grad, nlp_hessian_t& lag_hessian,
                       nlp_jacobian_t& A, Eigen::Ref<nlp_constraints_t> b) noexcept
    {
        static_cast<Derived*>(this)->update_linearisation_sparse_impl(x, p, x_step, lam, cost_grad, lag_hessian, A, b);
    }

  /** Hessian update: room for creativity: L-BFGS, BFGS (default), sparse BFGS, SR1 */
  EIGEN_STRONG_INLINE void hessian_update(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t>& x_step,
                                          const Eigen::Ref<const nlp_variable_t>& grad_step) noexcept
  {
      static_cast<Derived*>(this)->hessian_update_impl(hessian, x_step, grad_step);
  }

  /** termination criteria */
  EIGEN_STRONG_INLINE bool termination_criteria(const Eigen::Ref<const nlp_variable_t>& x) noexcept
  {
      return static_cast<Derived*>(this)->termination_criteria_impl(x);
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
                                              Eigen::Ref<nlp_jacobian_t> A, Eigen::Ref<nlp_constraints_t> b) noexcept
  {
      scalar_t _lag(0.0);
      problem.lagrangian_gradient_hessian(x,p,lam, _lag, m_lag_gradient, lag_hessian, cost_grad, b, A);
      hessian_regularisation(m_H);
  }

  // sparse version
  EIGEN_STRONG_INLINE void
  linearisation_sparse_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                            const Eigen::Ref<const nlp_dual_t>& lam,
                            Eigen::Ref<nlp_variable_t> cost_grad, nlp_hessian_t& lag_hessian,
                            nlp_jacobian_t& A, Eigen::Ref<nlp_constraints_t> b) noexcept
  {
      scalar_t _lag;
      problem.lagrangian_gradient_hessian(x,p,lam, _lag, m_lag_gradient, lag_hessian, cost_grad, b, A);
      hessian_regularisation(m_H);
  }

  /** default linearisation update uses damped BFGS algorithm */
  EIGEN_STRONG_INLINE void update_linearisation_dense_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                 const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                                 Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                 Eigen::Ref<nlp_jacobian_t> A, Eigen::Ref<nlp_constraints_t> b) noexcept;

  EIGEN_STRONG_INLINE void update_linearisation_sparse_impl(const Eigen::Ref<const nlp_variable_t>& x, const Eigen::Ref<const parameter_t>& p,
                                                            const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                                                            Eigen::Ref<nlp_variable_t> cost_grad, nlp_hessian_t& lag_hessian,
                                                            nlp_jacobian_t& A, Eigen::Ref<nlp_constraints_t> b) noexcept;

  /** default Hessain update -> dense damped BFGS */
  EIGEN_STRONG_INLINE void hessian_update_impl(Eigen::Ref<nlp_hessian_t> hessian, const Eigen::Ref<const nlp_variable_t>& x_step,
                                               const Eigen::Ref<const nlp_variable_t>& grad_step) noexcept
  {
        BFGS_update(hessian, x_step, grad_step);
  }

  EIGEN_STRONG_INLINE bool termination_criteria_impl(const Eigen::Ref<const nlp_variable_t>& x) noexcept;

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

template<typename Derived, typename Problem, typename QPSolver, typename Preconditioner>
typename SQPBase<Derived, Problem, QPSolver, Preconditioner>::scalar_t
SQPBase<Derived, Problem, QPSolver, Preconditioner>::step_size_selection_impl(const Eigen::Ref<const nlp_variable_t>& p) noexcept
{
    scalar_t mu, phi_l1, Dp_phi_l1;
    nlp_variable_t cost_gradient = this->m_h;
    const scalar_t tau = this->m_settings.tau; // line search step decrease, 0 < tau < settings.tau

    scalar_t constr_l1 = this->constraints_violation(this->m_x);

    // TODO: get mu from merit function model using hessian of Lagrangian
    //const scalar_t quad_term = p.dot(this->m_H * p);
    //const scalar_t qt = quad_term >= 0 ? scalar_t(0.5) * quad_term : 0;
    //mu = (abs(cost_gradient.dot(p)) ) / ((1 - this->m_settings.rho) * constr_l1);
    mu = this->m_lam_k.template lpNorm<Eigen::Infinity>();

    scalar_t cost_1;
    this->problem.cost(this->m_x, this->m_p, cost_1);

    phi_l1 = cost_1 + mu * constr_l1;
    Dp_phi_l1 = cost_gradient.dot(p) - mu * constr_l1;

    scalar_t alpha = scalar_t(1.0);
    scalar_t cost_step;
    nlp_variable_t x_step;
    for (int i = 1; i < this->m_settings.line_search_max_iter; i++)
    {
        x_step.noalias() = alpha * p;
        x_step += this->m_x;
        this->problem.cost(x_step, this->m_p, cost_step);
        m_cost = cost_step; //log cost

        scalar_t phi_l1_step = cost_step + mu * this->constraints_violation(x_step);

        if (phi_l1_step <= (phi_l1 + alpha * this->m_settings.eta * Dp_phi_l1))
            return alpha;
        else
            alpha = tau * alpha;
    }

    return alpha;
}

template<typename Derived, typename Problem, typename QPSolver, typename Preconditioner>
typename SQPBase<Derived, Problem, QPSolver, Preconditioner>::scalar_t
SQPBase<Derived, Problem, QPSolver, Preconditioner>::constraints_violation_impl(const Eigen::Ref<const nlp_variable_t>& x) const noexcept
{
    scalar_t cl1 = EPSILON;

    // c(x) = 0
    nlp_eq_constraints_t c;
    problem.equalities(x, m_p, c);
    cl1 += c.template lpNorm<1>();

    // l <= g(x) <= u
    nlp_ineq_constraints_t g;
    problem.inequalities(x, m_p, g);
    cl1 += (m_lbg - g).cwiseMax(0.0).sum();
    cl1 += (g - m_ubg).cwiseMax(0.0).sum();


    // l <= x <= u
    cl1 += (m_lbx - x).cwiseMax(0.0).sum();
    cl1 += (x - m_ubx).cwiseMax(0.0).sum();

    return cl1;
}

template<typename Derived, typename Problem, typename QPSolver, typename Preconditioner>
typename SQPBase<Derived, Problem, QPSolver, Preconditioner>::scalar_t
SQPBase<Derived, Problem, QPSolver, Preconditioner>::max_constraints_violation_impl(const Eigen::Ref<const nlp_variable_t>& x) const noexcept
{
    scalar_t c = scalar_t(0);

    // c_eq = 0
    if (NUM_EQ > 0)
    {
        nlp_eq_constraints_t c_eq;
        problem.equalities(x, m_p, c_eq);
        c = c_eq.template lpNorm<Eigen::Infinity>();
    }

    // lbg <= g <= ubg
    if (NUM_INEQ > 0)
    {
        nlp_ineq_constraints_t g;
        problem.inequalities(x, m_p, g);
        c = fmax( c, (m_lbg - g).maxCoeff() );
        c = fmax( c, (g - m_ubg).maxCoeff() );
    }

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

template<typename Derived, typename Problem, typename QPSolver, typename Preconditioner>
void SQPBase<Derived, Problem, QPSolver, Preconditioner>::update_linearisation_dense_impl(const Eigen::Ref<const nlp_variable_t>& x,
                                                                                          const Eigen::Ref<const parameter_t>& p,
                                                          const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                                                          Eigen::Ref<nlp_variable_t> cost_grad, Eigen::Ref<nlp_hessian_t> lag_hessian,
                                                          Eigen::Ref<nlp_jacobian_t> A, Eigen::Ref<nlp_constraints_t> b) noexcept
{
    scalar_t _lag;
    nlp_variable_t _lag_grad;
    problem.lagrangian_gradient(x, p, lam, _lag, _lag_grad, cost_grad, b, A);

    /** @badcode: redo with gradient step:  BFGS update */
    //std::cout << "optimality: " << _lag_grad.template lpNorm<Eigen::Infinity>() << "\n";
    hessian_update(lag_hessian, x_step, (_lag_grad - m_lag_gradient));
    m_lag_gradient = _lag_grad;
}

template<typename Derived, typename Problem, typename QPSolver, typename Preconditioner>
void SQPBase<Derived, Problem, QPSolver, Preconditioner>::update_linearisation_sparse_impl(const Eigen::Ref<const nlp_variable_t>& x,
                                                                                           const Eigen::Ref<const parameter_t>& p,
                                                          const Eigen::Ref<const nlp_variable_t>& x_step, const Eigen::Ref<const nlp_dual_t>& lam,
                                                          Eigen::Ref<nlp_variable_t> cost_grad, nlp_hessian_t& lag_hessian,
                                                          nlp_jacobian_t& A, Eigen::Ref<nlp_constraints_t> b) noexcept
{
    scalar_t _lag;
    nlp_variable_t _lag_grad;
    problem.lagrangian_gradient(x, p, lam, _lag, _lag_grad, cost_grad, b, A);

    /** @badcode: redo with gradient step:  BFGS update */
    //std::cout << "optimality: " << _lag_grad.template lpNorm<Eigen::Infinity>() << "\n";
    hessian_update(lag_hessian, x_step, (_lag_grad - m_lag_gradient));
    m_lag_gradient = _lag_grad;
}

template<typename Derived, typename Problem, typename QPSolver, typename Preconditioner>
bool SQPBase<Derived, Problem, QPSolver, Preconditioner>::termination_criteria_impl(const Eigen::Ref<const nlp_variable_t>& x) noexcept
{
    m_max_violation = max_constraints_violation(x);
    return (m_primal_norm <= m_settings.eps_prim) && (m_dual_norm <= m_settings.eps_dual) &&
            (m_max_violation <= m_settings.eps_prim) ? true : false;
}


template<typename Derived, typename Problem, typename QPSolver, typename Preconditioner>
void SQPBase<Derived, Problem, QPSolver, Preconditioner>::solve_qp(Eigen::Ref<nlp_variable_t> prim_step, Eigen::Ref<nlp_dual_t> dual_step) noexcept
{
    // solve the QP
    //typename qp_solver_t::status_t qp_status;
    status_t qp_status;

    /**
    std::cout << "H: \n" << m_H << "\n";
    std::cout << "A: \n" << m_A.template rightCols<29>() << "\n";
    std::cout << "h: " << m_h.transpose() << "\n";
    std::cout << "al: " << m_al.transpose() << "\n";
    std::cout << "ul: " << m_au.transpose() << "\n";
    std::cout << "xl: " << m_lx.transpose() << "\n";
    std::cout << "xu: " << m_ux.transpose() << "\n";
    */

    /** @badcode: m_x -> x step; m_lam -> lam step */
    qp_status = m_qp_solver.solve(m_H, m_h, m_A, m_al, m_au, m_lx, m_ux);
    //std::cout << "QP status: " << qp_status << " QP iter: " << m_qp_solver.info().iter << "\n";

    //eigen_assert(qp_status == status_t::SOLVED);

    m_info.qp_solver_iter += m_qp_solver.info().iter;

    prim_step = m_qp_solver.primal_solution();
    dual_step = m_qp_solver.dual_solution();

    //std::cout << "p: \n" << prim_step.transpose() << "\n";
    //std::cout << "l: \n" << dual_step.transpose() << "\n";

}

/** solve method */
template<typename Derived, typename Problem, typename QPSolver, typename Preconditioner>
void SQPBase<Derived, Problem, QPSolver, Preconditioner>::solve() noexcept
{
    m_info.status.value = sqp_status_t::MAX_ITER_EXCEEDED;

    nlp_variable_t p;   // search direction
    nlp_dual_t p_lambda; // dual search direction
    scalar_t alpha;    // step size

    p_lambda.setZero();

    // initialize
    m_info.qp_solver_iter = 0;
    m_info.iter = 1;

    linearisation(m_x, m_p, m_lam, m_h, m_H, m_A, m_al);
    /** place for heuristics: regularisation and preconditioning */

    /** prepare and scale the QP */
    /** @bug: just to note */
    m_al = -m_al;
    m_au =  m_al;
    m_al.template tail<NUM_INEQ>() += m_lbg;
    m_au.template tail<NUM_INEQ>() += m_ubg;
    m_lx.noalias() = m_lbx - m_x;
    m_ux.noalias() = m_ubx - m_x;

    /**
    std::cout << "H: \n" << m_H << "\n";
    std::cout << "A: \n" << m_A << "\n";
    std::cout << "h: " << m_h.transpose() << "\n";
    std::cout << "al: " << m_al.transpose() << "\n";
    std::cout << "ul: " << m_au.transpose() << "\n";
    std::cout << "xl: " << m_lx.transpose() << "\n";
    std::cout << "xu: " << m_ux.transpose() << "\n";
    */

    m_preconditioner.compute(m_H, m_h, m_A, m_al, m_au, m_lx, m_ux);

    /** solve and unscale the solution */
    solve_qp(p, p_lambda);
    m_preconditioner.unscale(p, p_lambda);
    //m_preconditioner.unscale_hessian(m_H);
    m_preconditioner.unscale(m_H, m_h, m_A, m_al, m_au, m_lx, m_ux);

    // in the case the sparse solver is used: reuse sparsity pattern for further solves
    m_qp_solver.settings().reuse_pattern = true;
    m_qp_solver.settings().warm_start    = true; // for other solvers

    m_lam_k = p_lambda;

    p_lambda -= m_lam;
    alpha = step_size_selection(p);

    //std::cout << "alpha: " << alpha << "\n";


    // take step
    m_x.noalias()   += alpha * p;
    m_lam.noalias() += alpha * p_lambda;

    // update step info
    m_step_prev.noalias() = alpha * p;
    m_primal_norm = alpha * p.template lpNorm<Eigen::Infinity>();
    m_dual_norm   = alpha * p_lambda.template lpNorm<Eigen::Infinity>();

    if (termination_criteria(m_x)) {
        m_info.status.value = sqp_status_t::SOLVED;
        return;
    }

    while (m_info.iter < m_settings.max_iter)
    {
        m_info.iter++;
        /** linearise and solve qp here*/
        //std::cout << "x:   " << m_x.transpose() << "\n";
        //std::cout << "dx:  " << m_step_prev.transpose() << "\n";
        //std::cout << "lam: " << m_lam.transpose() << "\n";

        //std::cout << "Hessian before: \n" << m_H << "\n";

        update_linearisation(m_x, m_p, m_step_prev, m_lam, m_h, m_H, m_A, m_al);

        /** prepare and scale the QP */
        m_al = -m_al;
        m_au =  m_al;
        m_al.template tail<NUM_INEQ>() += m_lbg;
        m_au.template tail<NUM_INEQ>() += m_ubg;
        m_lx.noalias() = m_lbx - m_x;
        m_ux.noalias() = m_ubx - m_x;

        //std::cout << "al: " << m_al.transpose() << "\n";
        //std::cout << "ul: " << m_au.transpose() << "\n";
        //std::cout << "xl: " << m_lx.transpose() << "\n";
        //std::cout << "xu: " << m_ux.transpose() << "\n";
        m_preconditioner.compute(m_H, m_h, m_A, m_al, m_au, m_lx, m_ux);

        solve_qp(p, p_lambda);
        m_preconditioner.unscale(p, p_lambda);
        m_preconditioner.unscale(m_H, m_h, m_A, m_al, m_au, m_lx, m_ux);
        //m_preconditioner.unscale_hessian(m_H);

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
