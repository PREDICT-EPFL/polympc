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




