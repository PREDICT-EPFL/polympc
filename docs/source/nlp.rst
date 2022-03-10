.. _chapter-nlp:

=====================
Nonlinear Programming
=====================

(Very) Brief Introduction
=========================

*PolyMPC* solves nonlinear optimisation programs (NLP) of the form:

.. math::

   \begin{equation}
   \begin{array}{ll}
    \underset{x}{\mbox{minimize}} & f(x) \\
    \mbox{subject to} & c(x) = 0 \\
                    & g_l \leq g(x) \leq g_u \\
                    & x_l \leq x \leq x_u
    \end{array}
    \end{equation}


with :math:`x \in \mathbb{R}^n`, equality constraint :math:`c(x): \mathbb{R}^n \to \mathbb{R}^m`, inequality constraint :math:`g(x): \mathbb{R}^n \to \mathbb{R}^g`,
and box constraint defined by vectors :math:`x_l \in \mathbb{R}^n` and :math:`x_u \in \mathbb{R}^n`. Before describing a numerical method for solving these NLPs
let's introduce basic definitions and concepts in nonlinear optimisation.

The Lagrangian for this NLP is defined as

.. math::

   \begin{equation}
   \begin{aligned}
   \mathcal{L}(x, \lambda_c, \lambda_g, \lambda_x) = f(x) + \lambda_c^Tc(x) + \lambda_g^Tg(x) + \lambda_x^Tx
   \end{aligned}
   \end{equation}

Where :math:`\lambda_c`, \lambda_g, \lambda_x` are the Lagrange multipliers corresponding to the constraints :math:`c(x) = 0`, :math:`g_l \leq g(x) \leq g_u`
and :math:`x_l \leq x \leq x_u` respectively.

The :math:`i-`th constraint :math:`g_{l_i} \leq g_i(x) \leq g_{u_i}` is called **inactive** if the inequalities hold strictly, otherwise it is called **active**.
A subset of of all active constraints will be denoted by a subscript :math:`\mathcal{A}`, e.g., :math:`g_\mathcal{A}`. A point :math:`x^{\star}` is called `regular` if it
is feasible and the following matrix has full row rank:

.. math::

   \begin{equation}
   \begin{aligned}
   \begin{bmatrix}
   \nabla_{x} c(x^{\star}) \\
   \nabla_{x} g(x^{\star})_{\mathcal{A}}
   \end{bmatrix}
   \end{aligned}
   \end{equation}

The last conditions are referred to as the linear independence constraint qualification (LICQ).

The Karush-Kuhn-Tucker conditions (KKT conditions) state the necessary conditions of optimality for an NLP: for a regular and locally optimal point :math:`x^{\star}` there
exist Lagrange multipliers :math:`\lambda_c^{\star}`, :math:`\lambda_g^{\star}`, :math:`\lambda_x^{\star}` such that the following holds:

- **Stationarity:** :math:`\begin{equation} \nabla_x \mathcal{L}(x^{\star}, \lambda_c^{\star}, \lambda_g^{\star}, \lambda_x^{\star}) = 0 \end{equation}`
- **Primal feasibility:** :math:`\begin{equation} c(x^{\star}) = 0, g_l \leq g(x^{\star}) \leq g_u, x_l \leq x^{\star} \leq x_u \end{equation}`
- **Dual feasibility and complimentarity:** :math:`(\lambda_g^{\star})_i \begin{cases} \geq 0 & \quad \text{if the i-th upper bound of} \ g(x) \  \text{is active} \\
  \leq 0 & \quad \text{if the i-th lower bound of}\  g(x)\  \text{is active} \\
  = 0 & \quad \text{if the i-th constraint is inactive}  \end{cases}`
  :math:`(\lambda_x^{\star})_i \begin{cases} \geq 0 & \quad \text{if the i-th upper bound of} \ x \  \text{is active} \\
  \leq 0 & \quad \text{if the i-th lower bound of} \ x \ \text{is active} \\
  = 0 & \quad \text{if the i-th box constraint is inactive}  \end{cases}`

Sequential Quadratic Programming
================================

The sequential quadratic programming (SQP) approach approximates the NLP problem by a constrained quadratic subproblem at the current
iteration :math:`x^k`, and the minimizer of this subproblem :math:`p^{\star}` is used to compute the next iterate :math:`x^{k+1}`. For equality-constrained NLPs,
solving this subproblem is equivalent to applying a Newton iteration to the KKT conditions mentioned above. A good reading about the topic is [Nocedal2006]_.

.. math::

   \begin{equation}
   \begin{array}{ll}
      \underset{p}{\mbox{minimize}} & \frac{1}{2} p^T H p + h^Tp \\
      \mbox{subject to} & c(x^k) + A_{c}p = 0 \\
                        & g_l - g(x^k) \leq A_{g}p \leq g(x^k) - g_u \\
                        & x_l - x^k \leq p \leq x_u - x^k
   \end{array}
   \end{equation}

with :math:`A_c \in \mathbb{R}^{m \times n}` and :math:`A_g \in \mathbb{R}^{g \times n}` the Jacobian of the constraints :math:`c(x)` and :math:`g(x)` at point :math:`x^k`,
the Hessian of the Lagrangian :math:`H = \nabla_{xx}L(x^k,\lambda^k)z, \lambda^k = [\lambda_c^{k}, \lambda_g^{k}, \lambda_x^k]^T` and objective gradient
:math:`h = \nabla_xf(x^k)`.

An extended version of the line search SQP algorithm from [Nocedal2006]_ is presented below. Here :math:`r_{prim}` denotes the maximum constraint
violation of the NLP at the iterate :math:`x^k`, and :math:`\alpha^0 = 1`.

1. **Given**: :math:`x^0, \lambda^0, p^0, p^0_\lambda, \ \epsilon_p, \epsilon_\lambda, \epsilon > 0`
2. **While** :math:`\alpha^k \Vert p^k \Vert_\infty \geq \epsilon_p \ || \  \alpha^k \Vert p_\lambda^k \Vert_\infty \geq \epsilon_\lambda \ || \ r_{prim} \geq \epsilon`
3. :math:`\quad` :math:`(H, h, c, A_c, g, A_g) \gets` linearization(:math:`x^k, \lambda^k`) of the nonlinear problem
4. :math:`\quad` :math:`H \gets` apply regularisation(*H*)
5. :math:`\quad` Apply preconditioning to QP subproblem (optional)
6. :math:`\quad` :math:`(p^k, \hat{\lambda}) \gets` solve QP subproblem
7. :math:`\quad` :math:`(p^k, \hat{\lambda}) \gets` revert preconditioning
8. :math:`\quad` :math:`p^k_{\lambda} \gets \hat{\lambda} - \lambda^k`
9. :math:`\quad` :math:`\alpha^k \gets` line search(:math:`p^k, \lambda^k`)
10. :math:`\quad` :math:`x^{k+1} \gets \alpha^k p^k` and :math:`\lambda^{k+1} \gets \alpha^k p^k_{\lambda}`
11. **End While**


Modelling NLP Problems
======================

Similar to the QP solvers, the user should specify the problem dimensions and data types at compile-time. This is done to enable embedded applications of the tool.

- ``Problem``: The user specifies an NLP by deriving from the :class:`ProblemBase` class. At this point the problem contains meta data such as dimensions, number of constraints,
  functions to evaluate cost and constraints. NLP solver will then use this data to actually allocate neccessary objects.

- ``QP Solver [Optional, Default: boxADMM]``: here any of the QP solvers described in :ref:`chapter-qp_methods` chapter can be used.

- ``QP Preconditioner [Optional, Default: IdentityPreconditioner]``: *ADMM* methods are known to be sensitive to the problem conditioning [Stellato2020]_, [Stathopoulos2016]_.
  However, it is often possible to scale the problem data to improve the conditioning and achieve better convergence of the QP solver. At the moment the user can choose
  between :class:`IdentityPreconditioner` (no preconditioning, zero overhead) and a heuristic Ruiz matrix equlibration algorithm [Ruiz2001]_.

**A Guiding Example**

Let us consider a simple nonlinear optimisation problem (Problem 71 from the Hock-Schittkowski problem collection) to illustrate the PolyMPC interface.

.. math::

   \begin{equation}
   \begin{split}
   &\min_{x \in \mathcal{R}^4} \;  x_1 x_4 (x_1 + x_2 + x_3) + x_3 \\
        &\begin{split}
        s.t. \quad & x_1^2 + x_2^2 + x_3^2 + x_4^2 = 40  \\
        & x_1 x_2 x_3 x_4 \geq 25
        \end{split}
   \end{split}
   \end{equation}

with the starting point :math:`x^0 = \begin{bmatrix}1 &  5 &  5 &  1\end{bmatrix}^T`.

To implement this problem in PolyMPC the user might want to write the code that looks like this:

.. code:: c++

   POLYMPC_FORWARD_NLP_DECLARATION(/*Name*/ HS071, /*NX*/ 4, /*NE*/1, /*NI*/1, /*NP*/0, /*Type*/double);

   using namespace Eigen;

   class HS071 : public ProblemBase<HS071>
   {
   public:
        Matrix<scalar_t, 4, 1> SOLUTION = {1.00000000, 4.74299963, 3.82114998, 1.37940829};

        template<typename T>
        inline void cost_impl(const Ref<const variable_t<T>>& x,
                              const Ref<const static_parameter_t>& p, T& cost) const noexcept
        {
            cost = x(0)*x(3)*(x(0) + x(1) + x(2)) + x(2);
            polympc::ignore_unused_var(p);
        }

        template<typename T>
        inline void inequality_constraints_impl(const Ref<const variable_t<T>>& x,
                                                const Ref<const static_parameter_t>& p,
                                                Eigen::Ref<ineq_constraint_t<T>> constraint) const noexcept
        {
            // 25 <= x^2 + y^2 <= Inf -> will set bounds later once the problem is instantiated
            constraint << x(0)*x(1)*x(2)*x(3);
            polympc::ignore_unused_var(p);
        }

        template<typename T>
        inline void equality_constraints_impl(const Ref<const variable_t<T>>& x,
                                              const Ref<const static_parameter_t>& p,
                                              Ref<eq_constraint_t<T>> constraint) const noexcept
        {
            // x(0)^2 + x(1)^2 + x(2)^2 + x(3)^2 == 40
            constraint << x.squaredNorm() - 40;
            polympc::ignore_unused_var(p);
        }
   };

Let us see closely what is going on.

.. code:: c++

   POLYMPC_FORWARD_NLP_DECLARATION(/*Name*/ HS071, /*NX*/ 4, /*NE*/1, /*NI*/1, /*NP*/0, /*Type*/double);

   using namespace Eigen;

This macro creates class traits for :class:`HS071` which allow to deduce compile information about the problem. The arguments here are: ``Name`` of the problem class
(should coincide with the later class declaration), ``NX``- number of optimisation variables, ``NE``- number of equality constraints, ``NI``- number of inequality
constraints, ``NP``- number of static problem parameters, ``Type``- scalar type. ``Using namespace Eigen`` is for here brevity and not encouraged in general.

.. NOTE::
   ``NP`` parameter is not neccessarily needed strictly speaking even if the problem has parameters that can be changed between solve. The user can simply make these
   parameters attributes of :class:`HS071` and use them similarly in problem formulation.


.. code:: c++

   class HS071 : public ProblemBase<HS071>
   {
   public:
   ...
   };

Here class :class:`HS071` inherits type aliases for optimisation variables, constraints and internal objects required by an optimisation algorithm: *gradient*,
constraints *Jacobian*, *Lagrangian*, *Hessian*, *dual variables*, etc. For dense problems, :class:`ProblemBase` also provides methods to evaluate sensitivities
using forward mode automatic differentiation. A summary of available types and (interface) methods is given below:

- :class:`scalar_t`: scalar type
- :class:`variable_t<T>`: parametric vector of optimisation variables
- :class:`constraint_t<T>`: parametric vector of equality and inequality constraints
- :class:`eq_constraint_t<T>`: parametric vector of equality constraints
- :class:`ineq_constraint_t<T>`: parametric vector of inequality constraints

.. NOTE::
   Class template parameter ``T`` is either a simple scalar or an automatic differentiation type depending on the circumstances.

The folowing types are non-parametric:

- :class:`nlp_variable_t`: vector optimisation variables
- :class:`nlp_eq_constraints_t`: vector of equality constraints
- :class:`nlp_ineq_constraints_t`: vector of inequality constraints
- :class:`nlp_constraints_t`: parametric vector constraints
- :class:`nlp_eq_jacobian_t`: equality constraints Jacobian
- :class:`nlp_ineq_jacobian_t`: inequality constraints Jacobian
- :class:`nlp_jacobian_t`: constraints Jacobian
- :class:`nlp_hessian_t`: Hessian
- :class:`nlp_cost_t = scalar_t`: cost
- :class:`nlp_dual_t`: vector of dual variables (Lagrange multipliers)
- :class:`static_parameter_t`: vector of static parameters

**Interface Functions** (Problem evaluators)

*Cost*

.. function:: EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, scalar_t &cost) noexcept

*Gradient of the cost*

.. function:: EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient) noexcept

*Gradient and Hessian of the cost function*

.. function:: EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, scalar_t &_cost, Eigen::Ref<nlp_variable_t> _cost_gradient, Eigen::Ref<nlp_hessian_t> hessian) noexcept

*Equalities*

.. function:: EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, Eigen::Ref<nlp_eq_constraints_t> _equalities) const noexcept

*Inequalities*

.. function:: EIGEN_STRONG_INLINE void inequalities(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, Eigen::Ref<nlp_ineq_constraints_t> _inequalities) const noexcept

*Constraints (inequalities and equalities combined)*

.. function:: EIGEN_STRONG_INLINE void constraints(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, Eigen::Ref<nlp_constraints_t> _constraints) const noexcept

*Linearised Equalities*

.. function:: EIGEN_STRONG_INLINE void equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, Eigen::Ref<nlp_eq_constraints_t> equalities, Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept

*Linearised Inequalities*

.. function:: EIGEN_STRONG_INLINE void inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, Eigen::Ref<nlp_ineq_constraints_t> inequalities, Eigen::Ref<nlp_ineq_jacobian_t> jacobian) noexcept

*Linearised Constraints*

.. function:: EIGEN_STRONG_INLINE void inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, Eigen::Ref<nlp_ineq_constraints_t> inequalities, Eigen::Ref<nlp_ineq_jacobian_t> jacobian) noexcept

*Lagrangian*

.. function:: EIGEN_STRONG_INLINE void lagrangian(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian) const noexcept

*Gradient of Lagrangian*

.. function:: EIGEN_STRONG_INLINE void lagrangian_gradient(const Eigen::Ref<const nlp_variable_t>& var, const Eigen::Ref<const static_parameter_t>& p, const Eigen::Ref<const nlp_dual_t>& lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> _lag_gradient) noexcept

*Gradient and Hessian of Lagrangian*

.. function:: EIGEN_STRONG_INLINE void lagrangian_gradient_hessian(const Eigen::Ref<const nlp_variable_t> &var, const Eigen::Ref<const static_parameter_t> &p, const Eigen::Ref<const nlp_dual_t> &lam, scalar_t &_lagrangian, Eigen::Ref<nlp_variable_t> lag_gradient, Eigen::Ref<nlp_hessian_t> lag_hessian, Eigen::Ref<nlp_variable_t> cost_gradient, Eigen::Ref<nlp_constraints_t> g, Eigen::Ref<nlp_jacobian_t> jac_g, const scalar_t cost_scale) noexcept

.. NOTE::
   Large number of arguments in sensitivity evaluation functions is explained by computations optimisation. Typically, when one needs to evaluate Hessian of the Lagrangian using
   automatic differentiation (operator overloading at least), gradient and Lagrangian values itself come as a by-product of computations. Let alone, cost gradient and constraints
   Jacoabian. This is why we prefer to evaluate and provide all these values.

**Example**

Assuming :class:`HS071` is defined as before, we can query the problem functions and their sensitivities (for **DENSE** problems) for some primal and dual query points.

.. code:: c++

   int main(void)
   {
        HS071 problem;
        HS071::nlp_variable_t x;
        HS071::nlp_dual_t y;
        HS071::static_parameter_t p;
        HS071::nlp_cost_t cost, lagrangian;
        HS071::nlp_variable_t lag_gradient, cost_gradient;
        HS071::nlp_hessian_t lag_hessian;
        HS071::nlp_jacobian_t jacobian;
        HS071::nlp_constraints_t constraints;

        /** Evaluate Lagrangian, it's gradient and Hessian for some query point*/
        x.setZero();
        y.setOnes();

        problem.lagrangian_gradient_hessian(x, p, y, lagrangian, lag_gradient, lag_hessian, cost_gradient, constraints, jacobian, 1.0);

        std::cout << "Hessian of Lagrangian: \n" << lag_hessian << "\n";

        return EXIT_SUCCESS;
    }

.. NOTE::
   The ``1.0`` scalar in the end of the function considered in the example is for compatibility with *Interior Point Methods* which require scaling of the cost function
   with respect to the "relaxed" constraints.

.. NOTE::
   Automatic sensitivities generation is available for **DENSE** problems only at the moment. Efficient sensitivities evaluation for **SPARSE** problems will require
   sparsity pattern estimation and is not implemented yet. It's not prohobited to create sparse Jacoabians and Hessians (as shown next) but will be less efficient than
   dense. The user is recommended to overload all sensitivity computation functions by implementing ``(function_name)_impl`` function with the same signature.

**Sparse Sensitivities Generation**

.. code:: c++

   class HS071 : public ProblemBase<HS071, SPARSE>
   {
   public:
   ...

   /** user defined evaluation of constraints and Jacobian */
   EIGEN_STRONG_INLINE void constraints_linearised_impl(const Eigen::Ref<const nlp_variable_t>& var,
                                                        const Eigen::Ref<const static_parameter_t>& p,
                                                        Eigen::Ref<nlp_constraints_t> constraints,
                                                        Eigen::Ref<nlp_jacobian_t> jacobian) noexcept
   {
        specific implementation
   }

   ...
   };

This class will generate interface functions to work with sparse Jacoabian and Hessians.


Solving NLP Problems
====================

Now that we can evaluate NLP problem functions and its sensitivities, let us finally solve the problem using our SQP method. First, we need to create a solver.
We will create a parametric solver (derives from :class:`SQPBase`) that can solve any NLP problem, for instance :class:`HS071`:

.. code:: c++

   template<typename Problem>
   class SQPSolver : public SQPBase<SQPSolver<Problem>, Problem>
   {

        /** it has no modifications so far- we will leave it for the next section */

   };

And solve:

.. code:: c++

    int main(void)
    {
        using Solver = SQPSolver<HS071>;
        HS071 problem;
        Solver solver;
        Solver::nlp_variable_t x0, x;
        Solver::nlp_dual_t y0;
        y0.setZero();
        x0 << 1.0, 5.0, 5.0, 1.0; // initial guess

        solver.settings().max_iter = 50;
        solver.settings().line_search_max_iter = 5;

        /** set inequalities bounds */
        solver.lower_bound_g() << 25;
        solver.upper_bound_g() << std::numeric_limits<Solver::scalar_t>::infinity();

        /** box constraints */
        solver.lower_bound_x() << 1.0, 1.0, 1.0, 1.0;
        solver.upper_bound_x() << 5.0, 5.0, 5.0, 5.0;
        solver.solve(x0, y0);

        x = solver.primal_solution();

        std::cout << "Number of iterations: " << solver.info().iter << std::endl;
        std::cout << "Solution: " << x.transpose() << std::endl;

        return EXIT_SUCCESS;
    }


Interface to IPOPT
==================

Besides the custom SQP implementation, PolyMPC provides an inteface to a well established interior point solver `IPOPT <https://coin-or.github.io/Ipopt/>`_.

.. code:: c++

   int main(void)
   {
       using Solver = IpoptInterface<HS071>;
       HS071 problem;
       Solver solver;

       // try to solve the problem
       solver.lower_bound_x() << 1.0, 1.0, 1.0, 1.0;
       solver.upper_bound_x() << 5.0, 5.0, 5.0, 5.0;
       solver.lower_bound_g() << 0.0, 25.0;
       solver.upper_bound_g() << 0.0, std::numeric_limits<double>::infinity();
       solver.primal_solution() << 1.0, 5.0, 5.0, 1.0;

       std::cout << "Solving HS071 \n";

       Solver::nlp_variable_t x0, x;
       Solver::nlp_dual_t y0;

       solver.solve();
       x = solver.primal_solution();

       return EXIT_SUCCESS;
   }

Solver Customisation and Settings
=================================

TODO...








