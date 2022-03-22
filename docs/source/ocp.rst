.. _chapter-ocp:

=========================
Nonlinear Optimal Control
=========================

Optimal control aims at finding a control function that minimises (or maximises) a given criterion subject to system dynamics, typically governed by differential equations,
and path constraints. A generic continuous-time Optimal Control Problem (OCP) is often written in the following form:

.. math::

   \begin{aligned}
   \underset{x(\cdot), u(\cdot), p}{\text{minimize}} \  & J[x(t_0),u(\cdot), p]  =  \mathit{\Phi[x(t_f), t_f, p]} + \int_{t_0}^{t_f} \mathit{L}[x(\tau), u(\tau), \tau, p] d\tau \\
   \mathrm{s.t.} \;\; & \forall \tau \in [t_0, t_f]  \\
   & \dot{x}(\tau) = f(x(\tau), u(\tau), \tau, p), \ x(t_0) = x_0 \\
   & g(x(\tau), u(\tau), \tau, p) \leq 0 \\
   & \psi(x(t_f), t_f, p) \leq 0
   \end{aligned}


where :math:`\tau \in \mathbb{R}` denotes time, :math:`x(\tau) \in \mathbb{R}^{n_x}` is the *state* of the system, :math:`u(\tau) \in \mathbb{R}^{n_u}` is the vector of *control* inputs
and :math:`p \in \mathbb{R}^{n_p}` is the vector of *parameters*. The function :math:`\mathit{\Phi} : \mathbb{R}^{n_x} \times \mathbb{R} \times \mathbb{R}^{n_p} \rightarrow \mathbb{R}`
is the *terminal cost function* and :math:`\mathit{L} : \mathbb{R}^{n_x} \times \mathbb{R}^{n_u} \times \mathbb{R} \times \mathbb{R}^{n_p}  \rightarrow \mathbb{R}` is called the
*running cost*. The *system dynamics* are given by the function :math:`f : \mathbb{R}^{n_x} \times \mathbb{R}^{n_u} \times \mathbb{R} \times \mathbb{R}^{n_p}  \rightarrow \mathbb{R}^{n_x}`,
:math:`g : \mathbb{R}^{n_x} \times \mathbb{R}^{n_u} \times \mathbb{R} \times \mathbb{R}^{n_p}  \rightarrow \mathbb{R}^{n_g}` is the *path constraint function*, and finally :math:`\psi :
\mathbb{R}^{n_x} \times \mathbb{R} \times \mathbb{R}^{n_p}  \rightarrow \mathbb{R}^{n_{\phi}}` is the terminal constraint function.


Numerical Methods
=================

*PolyMPC* is designed to solve OCPs of the form described above. Unlike other software tools for real-time optimal control, *PolyMPC* features optimised embeddable implementation of
*Chebyshev Pseudospectral Method*. Pseudospectral collocation is a numerical technique that employs a polynomial approximation of the state and control trajectories to
discretize continuous-time OCPs. Initially developed for solving differential equations, the collocation method was adopted for optimal control problems by Biegler [Biegler1984]_
and further analysed and developed by Ross [Fahroo2003]_,  Benson [Benson2005]_ and Huntington [Huntington2007]_ in the early 2000s. One distinctive feature of this
method is that the solution of a system of differential equations and optimisation of the cost function happen simultaneously. It also inherits the stability properties of the
collocation method for ODEs, and therefore is particularly useful for stiff nonlinear and unstable systems. Furthermore, for some smooth problems the method can exhibit very fast,
or spectral, convergence.

**Chebyshev Pseudospectral Method**

Assuming the approximate solution for the state and control trajectories are represented by an appropriately chosen set of basis functions :math:`\phi_{k}(\cdot)`:

.. math::

   \begin{equation}
   \begin{aligned}
   x(t) \approx x^{N}(t) = \sum_{k=0}^{N} x_{k} \phi_k(t) \\
   u(t) \approx u^{N}(t) = \sum_{k=0}^{N} u_{k} \phi_k(t) \\
   \end{aligned}
   \end{equation}

the collocation approach requires that the differential equation is satisfied exactly at a set of collocation points :math:`t_j \in (t_0, \ t_f)`:

.. math::

   \begin{equation}
   \begin{aligned}
   \left.\frac{dx^{N}}{dt} - f(x^{N}, u^{N}, t_j, p) \right \vert_{t=t_{j}} = 0 , \ j = 1,...,N \\
   \end{aligned}
   \end{equation}

With some initial conditions: :math:`x^{N}(t_0) = x_0`. Formulas represent a function expansion in the nodal basis, where :math:`x_k` is the value of :math:`x(t)` at the node
:math:`t_k` and :math:`\phi_{k}` denotes a specific cardinal basis function, which satisfies the condition: :math:`\phi_{j}(t_i) = \delta_{ij}, \ i,j = 0,...,N`, where:

.. math::

   \begin{equation}
   \begin{aligned}
   \delta_{ij} =
   \begin{cases}
   1 &  \ i = j \\
   0 & \ i \neq j
   \end{cases}
   \end{aligned}
   \end{equation}


A common choice of the cardinal basis in pseudospectral methods is the characteristic Lagrange polynomial of order :math:`N`. This means that each basis function reproduces one
function value at a particular point in the domain. The general expression for these polynomials is given by:

.. math::

   \begin{equation}
   \begin{aligned}
   \phi_k(t) = \prod^{N}_{j,\  j \neq k} \frac{(t - t_j)}{(t_k - t_j)}, \ k = 0,...,N \\
   \end{aligned}
   \end{equation}

Using the approximations above one can compute derivatives of the state trajectory at the nodal points:

.. math::

   \begin{equation}
   \begin{aligned}
   (x^{N}(t_j))' = \sum_{k=0}^{N} x_{k} \dot{\phi}_k(t_j) = \sum_{k=0}^{N} x_{k} D_{jk} \\
   \end{aligned}
   \end{equation}

where :math:`D_{jk}` are elements of the interpolation derivative matrix :math:`\mathbf{D}`. And the approximate solution to the differential equation satisfies a system of
algebraic equations:

.. math::

   \begin{equation}
   \begin{aligned}
   \sum_{k=0}^{N} x_{k} D_{jk} = f(x_j, u_j, t_j, p), \quad  j = 0,  \dots, N
   \end{aligned}
   \end{equation}

A careful choice of collocation points is crucial for numerical accuracy and convergence of the collocation method. An equidistant spacing of the collocation nodes is known
to cause an exponential error growth of the polynomial interpolation near the edges of the domain, an effect called Runge's phenomenon.

In the *PolyMPC* we employ the most commonly used set of :math:`N+1` Chebyshev Gauss-Lobatto (CGL) points:

.. math::

   \begin{equation}
   \begin{aligned}
   t_j = \cos \left( \frac{\pi j }{N} \right), \ j=0,...,N \\
   \end{aligned}
   \end{equation}

Using Chebyshev polynomials in the collocation scheme is not efficient because they do not satisfy the cardinality condition, and therefore
we will be using CGL nodes, which allow us to avoid Runge's phenomenon, in combination with Lagrange polynomials to solve the system differential equations.

**Cost Function**

The Mayer term of the cost function can be trivially approximated at the last collocation point:

.. math::

   \begin{equation}
   \begin{aligned}
   \mathit{\Phi[x(\tau_f)], \tau_f, p} = \mathit{\Phi[x_N, 1, p]} = \mathit{\Phi[x(1), 1, p]} \\
   \end{aligned}
   \end{equation}

For the Lagrange term approximation using the CGL set of points it is convenient to utilize the Clenshaw-Curtis quadrature integration scheme, as suggested
in [Trefethen2000]_. The method allows one to find a set of weights :math:`\{\omega_j\} \in \mathbb{R}, \ j = 0, \cdots , N` such that the following approximation
is exact for all polynomials :math:`p(t)` of order less than :math:`N` and corresponding weight function :math:`\omega(t)`:

.. math::

   \begin{equation}
   \begin{aligned}
   \sum_{j=0}^{N}p(t_j) \omega_j = \int_{-1}^{1}p(t)\omega(t)dt, \  p(t) \in \mathbb{P}_{N} \\
   \end{aligned}
   \end{equation}

Where :math:`\mathbb{P}_{N}` is the space of polynomials of degree less or equal than :math:`N` and :math:`p(t)` is a Chebyshev approximation of the integrated function:

.. math::

   \begin{equation}
   \begin{aligned}
   \mathit{L}(x(t), u(t), t, p) \approx p(t) = \sum_{k=0}^{N} a_k T_k(t) \\
   \end{aligned}
   \end{equation}

.. NOTE::

   The pseudospectral collocation method helps to transform the continous optimal control into a finite dimensional nonlinear optimisation problem (:ref:`chapter-nlp`)
   with respect to polynomial expansion coefficients. More details on the implementation of the numerical method and extension to spline can [Listov2020]_.

Modelling Optimal Control Problems
==================================

Following the *PolyMPC* concept, the user should specify the problem dimensions and data types at compile-time. This is done to enable embedded applications of the tool.

- ``Problem``: The user specifies an OCP by deriving from the :class:`ContinuousOCP` class. At this point the problem contains meta data such as dimensions, number of constraints,
  state and control data types, functions to evaluate cost and constraints. The derived :class:`Problem` class becomes a nonlinear problem and can passed to an NLP solver.

- ``Approximation``: This defines the control and state approximating function and numerical scheme. This is typically a :class:`Spline` class or :class:`Polynomial`.

- ``Sparsity [Optional (SPARSE/DENSE), Default: DENSE]``: This template argument defines whether sparse or dense representation of problem sensitivities should be used.


**A Guiding Example**

Let us demonstrate the functionality of software on a simple automatic parking problem. We will consider a simple mobile robot that should park at a specific point with a
specified orientation :math:`[0,0,0]` starting from some point :math:`(x,y,\theta)`, which is illustrated in the figure below.

.. image:: img/mobile_robot.png

The robot is controlled by setting front wheel velocity :math:`v` and the steering angle :math:`\phi`, :math:`d` denotes the wheel base of the robot. Assuming no wheel slip
and omitting dynamics, the robot differential equation can be written as:

.. math::

   \begin{equation}
   \begin{bmatrix}
   \dot{x} \\
   \dot{y} \\
   \dot{\theta}
   \end{bmatrix}=
   \begin{bmatrix}
   v \cos(\theta) \cos(\phi) \\
   v \sin(\theta) \cos(\phi) \\
   v \sin(\phi) / d
   \end{bmatrix}
   \end{equation}

One option to drive the robot to a desired spot is to penalise the squared distance to the target in the cost function. Below, we show how to formulate and solve such problem
using *PolyMPC*.

.. code:: c++

    #include "polynomials/ebyshev.hpp"
    #include "polynomials/splines.hpp"
    #include "control/continuous_ocp.hpp"
    #include "utils/helpers.hpp"

    #include <iomanip>
    #include <iostream>
    #include <chrono>

    #define POLY_ORDER 5
    #define NUM_SEG    2

    /** benchmark the new collocation class */
    using Polynomial = polympc::Chebyshev<POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
    using Approximation = polympc::Spline<Polynomial, NUM_SEG>;

    POLYMPC_FORWARD_DECLARATION(/*Name*/ RobotOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 0, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)

    using namespace Eigen;

    class RobotOCP : public ContinuousOCP<RobotOCP, Approximation, SPARSE>
    {
    public:
        ~RobotOCP() = default;

        RobotOCP()
        {
            /** initialise weight matrices to identity (for example)*/
            Q.setIdentity();
            R.setIdentity();
            QN.setIdentity();
        }

        Matrix<scalar_t, 3, 3> Q;
        Matrix<scalar_t, 2, 2> R;
        Matrix<scalar_t, 3, 3> QN;

        template<typename T>
        inline void dynamics_impl(const Ref<const state_t<T>> x, const Ref<const control_t<T>> u,
                                  const Ref<const parameter_t<T>> p, const Ref<const static_parameter_t> &d,
                                  const T &t, Ref<state_t<T>> xdot) const noexcept
        {
            polympc::ignore_unused_var(p);
            polympc::ignore_unused_var(t);

            xdot(0) = u(0) * cos(x(2)) * cos(u(1));
            xdot(1) = u(0) * sin(x(2)) * cos(u(1));
            xdot(2) = u(0) * sin(u(1)) / d(0);
        }

        template<typename T>
        inline void lagrange_term_impl(const Ref<const state_t<T>> x, const Ref<const control_t<T>> u,
                                       const Ref<const parameter_t<T>> p, const Ref<const static_parameter_t> d,
                                       const scalar_t &t, T &lagrange) noexcept
        {
            polympc::ignore_unused_var(p);
            polympc::ignore_unused_var(t);
            polympc::ignore_unused_var(d);

            lagrange = x.dot(Q.template cast<T>() * x) + u.dot(R.template cast<T>() * u);
        }

        template<typename T>
        inline void mayer_term_impl(const Ref<const state_t<T>> x, const Ref<const control_t<T>> u,
                                    const Ref<const parameter_t<T>> p, const Ref<const static_parameter_t> d,
                                    const scalar_t &t, T &mayer) noexcept
        {
            polympc::ignore_unused_var(p);
            polympc::ignore_unused_var(t);
            polympc::ignore_unused_var(d);
            polympc::ignore_unused_var(u);

            mayer = x.dot(QN.template cast<T>() * x);
        }
    };

Below we will look into the code in more detail.

.. code:: c++

    #include "polynomials/ebyshev.hpp"
    #include "polynomials/splines.hpp"
    #include "control/continuous_ocp.hpp"
    #include "utils/helpers.hpp"

These headers contain classes necessary to define the approximation (Chebyshev polynomials, splines) and :class:`ContinuousOCP`. The header *utils/helpers.hpp* constains
some useful utilities.


.. code:: c++

    #define POLY_ORDER 5
    #define NUM_SEG    2

    using Polynomial = polympc::Chebyshev<POLY_ORDER, polympc::GAUSS_LOBATTO, double>;
    using Approximation = polympc::Spline<Polynomial, NUM_SEG>;

    POLYMPC_FORWARD_DECLARATION(/*Name*/ RobotOCP, /*NX*/ 3, /*NU*/ 2, /*NP*/ 0, /*ND*/ 1, /*NG*/0, /*TYPE*/ double)

Here, we choose the parameters of the approximation: two-segement spline with Lagrange polynomials of order `5` in each segment. Furthermore, ``polympc::GAUSS_LOBATTO``
with :class:`Chebyshev` class means that we will be using Chebyshev-Gauss-Lobatto collocation scheme. Next, similar to nonlinear programs, macro :class:`POLYMPC_FORWARD_DECLARATION`
creates class traits for the class :class:`RobotOCP` to infer compile time information. Here, ``NX`` is again the dimension of the state space, ``NU``- control, ``NP``- variable
parameters, ``ND``- static parameters, ``NG``- number of constraints, and finally the scalar type (``double`` in this case).

.. NOTE::

   The ``NP`` parameters, unlike static ``ND`` parameters can be changed by the nonlinear solver during iterations. ``ND`` parameters can be changed by the user inbetween solver
   calls. ``ND`` parameters are reduntant, strictly speaking, as mutable attributes of :class:`RobotOCP` can be used instead; this feature is primarily exists for compatibility of
   `Eigen` and `CasADi` interfaces.

Let's now move on to defining the dynamics functions.

.. code:: c++

    class RobotOCP : public ContinuousOCP<RobotOCP, Approximation, SPARSE>
    {
    public:
        ~RobotOCP() = default;

        RobotOCP()
        {
            /** initialise weight matrices to identity (for example)*/
            Q.setIdentity();
            R.setIdentity();
            QN.setIdentity();
        }

        Matrix<scalar_t, 3, 3> Q;
        Matrix<scalar_t, 2, 2> R;
        Matrix<scalar_t, 3, 3> QN;

        template<typename T>
        inline void dynamics_impl(const Ref<const state_t<T>> x, const Ref<const control_t<T>> u,
                                  const Ref<const parameter_t<T>> p, const Ref<const static_parameter_t> &d,
                                  const T &t, Ref<state_t<T>> xdot) const noexcept
        {
            polympc::ignore_unused_var(p);
            polympc::ignore_unused_var(t);

            xdot(0) = u(0) * cos(x(2)) * cos(u(1));
            xdot(1) = u(0) * sin(x(2)) * cos(u(1));
            xdot(2) = u(0) * sin(u(1)) / d(0);
        }

Here, :class:`ContinuousOCP` creates templated types for state (:class:`state_t`), control (:class:`control_t`), parameters (:class:`parameter_t`), derivatives (:class:`state_t`).
Static parameter type is not templated. Templates are necesary to propagate special :class:`AutoDiffScalar` type objects through the user-defined functions in order to compute
sensitivities. The user should implement :func:`dynamics_impl` function with a given particular signature. Function :func:`ignore_unused_var` is here merely to suppress compiler
warnings about unused variables, it does not add any computational overhead.

As a next step we will define a cost that penalises the squared distance of the robot from the target, which may look like one below:

.. math::

   \begin{aligned}
   J[x(t_0),u(\cdot), p]  =  x^T(t_f) QN x(t_f) + \int_{t_0}^{t_f} x^T(\tau) Q x(\tau) + u^T(\tau) R u(\tau) d\tau \\
   \end{aligned}

In the code, the user has to implement Lagrange (:func:`lagrange_term_impl`) and Mayer (:func:`mayer_term_impl`) terms. By default, :math:`t_0= 0.0` and :math:`t_f = 1.0`, we
will explain in the next section how to chenge these values.

.. code:: c++

    template<typename T>
    inline void lagrange_term_impl(const Ref<const state_t<T>> x, const Ref<const control_t<T>> u,
                                   const Ref<const parameter_t<T>> p, const Ref<const static_parameter_t> d,
                                   const scalar_t &t, T &lagrange) noexcept
    {
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(d);

        lagrange = x.dot(Q.template cast<T>() * x) + u.dot(R.template cast<T>() * u);
    }

    template<typename T>
    inline void mayer_term_impl(const Ref<const state_t<T>> x, const Ref<const control_t<T>> u,
                                const Ref<const parameter_t<T>> p, const Ref<const static_parameter_t> d,
                                const scalar_t &t, T &mayer) noexcept
    {
        polympc::ignore_unused_var(p);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(u);

        mayer = x.dot(QN.template cast<T>() * x);
     }


.. NOTE::

   Inconvenience of using :func:`template cast<T>()` has to do with certain current limitations of forward-mode automatic differentiation code in Eigen. In case, the exact
   second order derivatives are not required by the nonlinear solver, i.e. BFGS approximation is used, the :func:`cast` function is not required.


**Generic Inequality Constraints**


Additionally, we could limit angular acceleration of the robot a bit. This heuristicaly can be achieved by introducing a nonlinear inequality constraint:

.. math::

   \begin{aligned}
   g_l \leq v^2 \cos(\phi) \leq g_u \\
   \end{aligned}

The user can implement this inequality constraint function in :class:`RobotOCP` as shown below:

.. code:: c++

    template<typename T>
    inline void inequality_constraints_impl(const Ref<const state_t<T>> x, const Ref<const control_t<T>> u,
                                            const Ref<const parameter_t<T>> p, const Ref<const static_parameter_t> d,
                                            const scalar_t &t, Eigen::Ref<constraint_t<T>> g) const noexcept
    {
        g(0) = u(0) * u(0) * cos(u(1));

        polympc::ignore_unused_var(x);
        polympc::ignore_unused_var(d);
        polympc::ignore_unused_var(t);
        polympc::ignore_unused_var(p);
    }

Values :math:`g_l` and :math:`g_u` will be set after in the solver.


Solving OCP
===========

Now :class:`RobotOCP` is in principle equivalent to a nonlinear program, i.e. the user can access cost, constraints, Lagrangian, their derivatives etc. We can proceed solving
this problem as described in :ref:`chapter-nlp` chapter. We create a nonlinear solver, set up settings.

.. code:: c++

    using box_admm_solver = boxADMM<RobotOCP::VAR_SIZE, RobotOCP::NUM_EQ, RobotOCP::scalar_t,
                                    RobotOCP::MATRIXFMT, linear_solver_traits<RobotOCP::MATRIXFMT>::default_solver>;

    using preconditioner_t = RuizEquilibration<RobotOCP::scalar_t, RobotOCP::VAR_SIZE,
                                               RobotOCP::NUM_EQ, RobotOCP::MATRIXFMT>;

    int main(void)
    {
        Solver<RobotOCP, box_admm_solver, preconditioner_t> solver;
        /** change the final time */
        solver.get_problem().set_time_limits(0, 2);

        /** set optimiser properties */
        solver.settings().max_iter = 10;
        solver.settings().line_search_max_iter = 10;
        solver.qp_settings().max_iter = 1000;

        /** set the parameter 'd' */
        solver.parameters()(0) = 2.0;

        /** initial conditions and constraints on the control signal */
        Eigen::Matrix<RobotOCP::scalar_t, 3, 1> init_cond; init_cond << 0.5, 0.5, 0.5;
        Eigen::Matrix<RobotOCP::scalar_t, 2, 1> ub; ub <<  1.5,  0.75;
        Eigen::Matrix<RobotOCP::scalar_t, 2, 1> lb; lb << -1.5, -0.75;

        /** setting up bounds and initial conditions: magic for now */
        solver.upper_bound_x().tail(22) = ub.replicate(11, 1);
        solver.lower_bound_x().tail(22) = lb.replicate(11, 1);

        solver.upper_bound_x().segment(30, 3) = init_cond;
        solver.lower_bound_x().segment(30, 3) = init_cond;

        /** solve the NLP */
        polympc::time_point start = polympc::get_time();
        solver.solve();
        polympc::time_point stop = polympc::get_time();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        /** retrieve solution and some statistics */
        std::cout << "Solve status: " << solver.info().status.value << "\n";
        std::cout << "Num iterations: " << solver.info().iter << "\n";
        std::cout << "Primal residual: " << solver.primal_norm() << " | dual residual: " << solver.dual_norm()
                  << " | constraints  violation: " << solver.constr_violation() << " | cost: " << solver.cost() <<"\n";
        std::cout << "Num of QP iter: " << solver.info().qp_solver_iter << "\n";
        std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";
        std::cout << "Size of the solver: " << sizeof (solver) << "\n";
        std::cout << "Solution: " << solver.primal_solution().transpose() << "\n";

        return EXIT_SUCCESS;
    }


MPC Wrapper
===========

Even though we can solve the OCP now, the user or control engineer is required to understand how the collocation method works and how control and state polynomial coefficients
are stored in memory which is not user friendly to say the least. Therefore, *PolyMPC* includes the :class:`MPC` wrapper class that abstracts the numerical scheme from the user.
Additionally, since the control and state trajectories are represented by splines it is convenient to evaluate the solution at any point, or interpolate them.

Here is how our particular example will look like with the MPC wrapper:

.. code:: c++

    int main(void)
    {
        /** create an MPC algorithm and set the prediction horison */
        using mpc_t = MPC<RobotOCP, Solver, box_admm_solver>;
        mpc_t mpc;
        mpc.settings().max_iter = 20;
        mpc.settings().line_search_max_iter = 10;
        mpc.set_time_limits(0, 2);

        /** problem data */
        mpc_t::static_param p; p << 2.0;          // robot wheel base
        mpc_t::state_t x0; x0 << 0.5, 0.5, 0.5;   // initial condition
        mpc_t::control_t lbu; lbu << -1.5, -0.75; // lower bound on control
        mpc_t::control_t ubu; ubu <<  1.5,  0.75; // upper bound on control

        mpc.set_static_parameters(p);
        mpc.control_bounds(lbu, ubu);
        mpc.initial_conditions(x0);

        /** solve */
        polympc::time_point start = polympc::get_time();
        mpc.solve();
        polympc::time_point stop = polympc::get_time();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        /** retrieve solution and statistics */
        std::cout << "MPC status: " << mpc.info().status.value << "\n";
        std::cout << "Num iterations: " << mpc.info().iter << "\n";
        std::cout << "Solve time: " << std::setprecision(9) << static_cast<double>(duration.count()) << "[mc] \n";

        std::cout << "Solution X: " << mpc.solution_x().transpose() << "\n";
        std::cout << "Solution U: " << mpc.solution_u().transpose() << "\n";

        /** sample x solution at collocation points [0, 5, 10] */
        std::cout << "x[0]: " << mpc.solution_x_at(0).transpose() << "\n";
        std::cout << "x[5]: " << mpc.solution_x_at(5).transpose() << "\n";
        std::cout << "x[10]: " << mpc.solution_x_at(10).transpose() << "\n";

        std::cout << " ------------------------------------------------ \n";

        /** sample control at collocation points */
        std::cout << "u[0]: " << mpc.solution_u_at(0).transpose() << "\n";
        std::cout << "u[1]: " << mpc.solution_u_at(1).transpose() << "\n";

        std::cout << " ------------------------------------------------ \n";

        /** sample state at time 't = [0.0, 0.5]' */
        std::cout << "x(0.0): " << mpc.solution_x_at(0.0).transpose() << "\n";
        std::cout << "x(0.5): " << mpc.solution_x_at(0.5).transpose() << "\n";

        std::cout << " ------------------------------------------------ \n";

        /**  sample control at time 't = [0.0, 0.5]' */
        std::cout << "u(0.0): " << mpc.solution_u_at(0.0).transpose() << "\n";
        std::cout << "u(0.5): " << mpc.solution_u_at(0.5).transpose() << "\n";

        return EXIT_SUCCESS;
    }


Below is the list of all currently available interface functions:


*Set Time Limits*

.. function:: inline void set_time_limits(const scalar_t& t0, const scalar_t& tf) noexcept

*Set Initial Conditions*

.. function:: inline void initial_conditions(const Eigen::Ref<const state_t>& x0) noexcept
.. function::  inline void initial_conditions(const Eigen::Ref<const state_t>& x0_lb, const Eigen::Ref<const state_t>& x0_ub) noexcept

*Set State (Trajectory) Bounds*

.. function:: void x_lower_bound(const Eigen::Ref<const state_t>& xlb)

.. function:: void x_upper_bound(const Eigen::Ref<const state_t>& xub)

.. function:: void state_bounds(const Eigen::Ref<const state_t>& xlb, const Eigen::Ref<const state_t>& xub)

.. function:: void state_trajectory_bounds(const Eigen::Ref<const traj_state_t>& xlb, const Eigen::Ref<const traj_state_t>& xub)

.. function:: void x_final_lower_bound(const Eigen::Ref<const state_t>& xlb)

.. function:: void x_final_upper_bound(const Eigen::Ref<const state_t>& xub)

.. function:: void final_state_bounds(const Eigen::Ref<const state_t>& xlb, const Eigen::Ref<const state_t>& xub)


*Set Control (Trajectory) Bounds*

.. function:: void u_lower_bound(const Eigen::Ref<const control_t>& lb)

.. function:: void u_upper_bound(const Eigen::Ref<const control_t>& ub)

.. function:: void control_trajecotry_bounds(const Eigen::Ref<const traj_control_t>& lb, const Eigen::Ref<const traj_control_t>& ub)

.. function:: void control_bounds(const Eigen::Ref<const control_t>& lb, const Eigen::Ref<const control_t>& ub)


*Set Generic Inequalities Bounds*

.. function:: void constraints_trajectory_bounds(const Eigen::Ref<const constraints_t>& lbg, const Eigen::Ref<const constraints_t>& ubg)

.. function:: void constraints_bounds(const Eigen::Ref<const constraint_t>& lbg, const Eigen::Ref<const constraint_t>& ubg)

*Set Variable Parameters Bounds*

.. function:: void parameters_bounds(const Eigen::Ref<const parameter_t>& lbp, const Eigen::Ref<const parameter_t>& ubp)

*Set Static Parameters*

.. function:: void set_static_parameters(const Eigen::Ref<const static_param>& param)

*Set State Trajectory Guess (for optimiser)*

.. function:: void x_guess(const Eigen::Ref<const traj_state_t>& x_guess)

*Set Control Trajectory Guess (for optimiser)*

.. function:: void u_guess(const Eigen::Ref<const traj_control_t>& u_guess)

*Set Parameters Guess*

.. function::  void p_guess(const Eigen::Ref<const parameter_t>& p_guess)

*Set Dual Variable Guess (if available)*

.. function:: void lam_guess(const Eigen::Ref<const dual_var_t>& lam_guess)

*Set/Get NLP Solver Settings*

.. function:: const typename nlp_solver_t::nlp_settings_t& settings() const

.. function:: typename nlp_solver_t::nlp_settings_t& settings()

*Set/Get QP Solver Settings*

.. function:: const typename nlp_solver_t::nlp_settings_t& settings() const

.. function:: typename nlp_solver_t::nlp_settings_t& settings()

*Access NLP Solver Info*

.. function:: const typename nlp_solver_t::nlp_info_t& info() const

.. function:: typename nlp_solver_t::nlp_info_t& info()

*Access NLP Solver (object)*

.. function:: const nlp_solver_t& solver() const

.. function:: nlp_solver_t& solver()

*Access OCP class (object)*

.. function:: const OCP& ocp() const

.. function:: OCP& ocp() noexcept

*Access NLP Solver Convergence Properties*

.. function const scalar_t primal_norm() const

.. function const scalar_t dual_norm()   const

.. function const scalar_t constr_violation() const

.. function const scalar_t cost() const


*Get State Trajectory as Column or [NX x NUM_NODES] matrix*

.. function:: traj_state_t solution_x() const

.. function:: Eigen::Matrix<scalar_t, nx, num_nodes> solution_x_reshaped() const

*Get State Trajectory at K-th Collocation Point*

.. function:: state_t solution_x_at(const int &k) const

*Get State Trajectory at Time Point 't' (interpolated)*

.. function:: state_t solution_x_at(const scalar_t& t) const


*Get Control Trajectory as Column or [NU x NUM_NODES] matrix*

.. function:: traj_control_t solution_u() const

.. function:: Eigen::Matrix<scalar_t, nu, num_nodes> solution_u_reshaped() const

*Get Control Trajectory at K-th Collocation Point*

.. function:: control_t solution_u_at(const int &k) const

*Get Control Trajectory at Time Point 't' (interpolated)*

.. function:: control_t solution_u_at(const scalar_t& t) const

*Get Optimal Parameters*

.. function:: parameter_t solution_p() const

*Get Dual Solution*

.. function:: dual_var_t solution_dual() const















