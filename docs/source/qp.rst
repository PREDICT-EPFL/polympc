.. _chapter-qp_methods:

=====================
Quadratic Programming
=====================

PolyMPC solves quadratic programs with two-sided linear inequality constraints of the form.

.. math::
   \begin{equation}
   \begin{array}{ll}
   \underset{x}{\text{minimize}}  & \frac{1}{2} x^T P x + q^T x \\
   \mbox{subject to} & l \leq A x \leq u \\
   & x_l \leq x \leq x_u
   \end{array}
   \end{equation}

where :math:`x \in R^n` is an optimisation variable, :math:`x_l  \in \mathbb{R}^n \cup -\infty` and :math:`x_u  \in \mathbb{R}^n \cup +\infty` are box constraints,
:math:`P \in \mathbb{S}^n_+` positive semidefinite,
:math:`q \in \mathbb{R}^n`,
:math:`A \in \mathbb{R}^{m \times n}`,
:math:`l \in \mathbb{R}^m \cup -\infty` and
:math:`u \in \mathbb{R}^m \cup +\infty`.
For equality constraints :math:`l_i = u_i`.

The software tool is originally designed for real-time embedded control of robotic and mechatronic systems. This design requirements motivates the choice of algorithms
for QP: we prefer methods with lower cost per iteration and fewer linear system factorisations which often suitable for problems where moderate accuracy is sufficient.

.. _section-admm:

Numerical Methods
=================

**Alternating Direction Methods of Multipliers (ADMM)**

The ADMM algorithm was chosen for its cheap and well vectorisable iterations that are particularly suitable for embedded applications. We first present a standard splitting
approach suggested in the literature [Boyd2011]_, [Stellato2020]_, [Stathopoulos2016]_. The key ingredients of the method (similar to other operator splitting methods):

* **Splitting step**. Assume for brevity that box contraints are part of the general polytopic constraints :math:`Ax`; here the idea is to "separate" the uncontrained quadratic
  cost function from the inequality constraints by introducing an auxillary variable :math:`z`:

.. math::
   \begin{equation}
   \begin{array}{ll}
   \underset{x, z}{\text{minimize}} & \frac{1}{2} x^T P x + q^T x \\
   \mbox{subject to} & l \leq z \leq u \\
   & Ax = z
   \end{array}
   \end{equation}

* **Equivalent QP**. Arguably, the main complexity of quadratic programming is handling inequality constraints which gives rise to various solving techniques. In splitting methods,
  we formulate an equivalent "easy" equality-constrained QP by introducing indicator functions to the cost:

.. math::
   \begin{equation}
   \begin{array}{ll}
   \underset{x, z, \tilde{x}, \tilde{z}}{\text{minimize}} & \frac{1}{2} \tilde{x}^T P \tilde{x} + q^T \tilde{x} + \mathcal{I}_{Ax=z}(\tilde{x},\tilde{z}) + \mathcal{I_C}(z) \\
   \mbox{subject to} & (\tilde{x},\tilde{z}) = (x,z)
   \end{array}
   \end{equation}


where :math:`x \in \mathbb{R}^n`, :math:`z \in \mathbb{R}^{m + n}`. :math:`\mathcal{I}_{\tilde{A}x=z}` is the indicator function for set :math:`\{(x,z) \mid \tilde{A}x = z \}` to enforce
:math:`\tilde{A}\tilde{x}=\tilde{z}` and :math:`\mathcal{I_C}` is the indicator function for set :math:`\mathcal{C} = [l,u] = \{z \mid l_i \leq z_i \leq u_i, i= 1 \cdots m+n \}` to enforce
:math:`l \leq z \leq u`.

* **Augmented Lagrangian** for this equivalent problem has the following form (after some algebraic manipulations):

.. math::
   \begin{equation}
   \begin{array}{ll}
   L_{\sigma\rho}(x,z,\tilde{x},\tilde{z},w,y)
   =& \frac{1}{2} \tilde{x}^T P \tilde{x} + q^T \tilde{x}
   + \mathcal{I}_{Ax=z}(\tilde{x},\tilde{z})
   + \mathcal{I_C}(z) \\
   &+ \frac{\sigma}{2}\Vert \tilde{x} - x + \sigma^{-1}w \Vert_{2}^2
   + \frac{\rho}{2}\Vert \tilde{z} - z + \rho^{-1}y |Vert_{2}^2
   \end{array}
   \end{equation}

with dual variables :math:`(w,y)` and corresponding penalty parameters :math:`(\sigma,\rho) > 0`.

* **Alternating Minimisation**. Finally, as the name suggests, the method minimises the Augmented Lagrangian by alternating between variables :math:`x,\tilde{x},z, \tilde{z}`
  (and the dual varables :math:`w,y`). Using Augmented Lagrangian has several benefits. First, it relaxes the requirement of strict convexity of the quadratic cost, therefore
  allowing for a positive semi-definite cost. Second, the operator splitting theory provides efficient expressions for evaluating certain proximal operators. In our case, the
  proximal operator for indicator funtions is a simple box projection. The basic ADMM interations, therefore, can be summarised as below:


.. math::
   \begin{align}
   (\tilde{x}^{k+1},\tilde{z}^{k+1}) &\gets
   \underset{(\tilde{x},\tilde{z}):\tilde{A}\tilde{x}=\tilde{z}}{\mathrm{argmin}}
   \frac{1}{2} \tilde{x}^T P \tilde{x} + q^T \tilde{x}
   % + \mathcal{I}_{Ax=z}(\tilde{x},\tilde{z})
   + \frac{\sigma}{2}\Vert \tilde{x} - x^k + \sigma^{-1}w^k \Vert_{2}^2
   + \frac{\rho}{2} \Vert \tilde{z} - z^k + \rho^{-1}y^k \Vert_{2}^2 \\
   x^{k+1} &\gets
   \tilde{x}^{k+1} + \sigma^{-1}w^k \\
   z^{k+1} &\gets
   \Pi_\mathcal{C}
   \left (
   \tilde{z}^{k+1} + \rho^{-1}y^k \right ) \\
   w^{k+1} &\gets
   w^k + \sigma(\tilde{x}^{k+1} - x^{k+1}) \\
   y^{k+1} &\gets
   y^k + \rho(\tilde{z}^{k+1} - z^{k+1})
   \end{align}


The method iterates until the primal and dual residuals satisfy user-specified tolerances: :math:`r_{prim} = \Vert Ax - z \Vert_\infty`
and :math:`r_{dual} = \Vert Px + q + A^Ty \Vert_\infty`


**boxADMM**

For dense problems with box constraints the splitting described before is not very efficient due to the potentially large number of zero
values in :math:`A` and the linear system at the step (1) in ADMM iterations that have to be stored. We therefore implement
an additional splitting which aims at reducing memory consumption and better vectorisation -- *boxADMM*. The method treats general polytopic
constraints and box differently which results in better performance especially for dense QPs.


Interfacing Other Solvers
=========================

PolyMPC provides a unified interface to some established codes for solving QP problems.

**OSQP**

`OSQP <https://osqp.org/>`_ is accessed via `osqp-eigen <https://github.com/robotology/osqp-eigen>`_ interface. *OSQP* is based on the ADMM framework
described in :ref:`section-admm` and features additionally infeasibility detection and solution polishing (in case more accurate solution is required).

.. NOTE::
   *OSQP* solver supports computations with **sparse** matrices only. Attempt to create a dense *OSQP* solver interface will result in a compilation error. The user
   is free to convert dense matrices to sparse using standard *Eigen* methods before passing them to the solver.

   When vectorisation is enabled the user can expected better performance from *ADMM* implementation in *PolyMPC*. Further, *OSQP* only supports allocation free
   operation in code-generation mode which is not available through the *PolyMPC* interface.


**QPMAD**

`QPMAD <https://github.com/asherikov/qpmad>`_ is a header-only *Eigen*-based implementation of the Goldfarb-Idnani dual active set method [Goldfarb1983]_.

.. NOTE::
   *QPMAD* solver supports only **dense** matrices. Attempt to create a sparse *QPMAD* solver interface will result in a compilation error.




Modelling Quadratic Programs
============================

To solve a QP with *PolyMPC* you need to specify several details about the problem at the compile-time. This is necessary for memory management and performance optimisation.

- ``Problem Dimensions``: The user need to specify the number of optmisation variables ``N`` and number of generic linear constraints (including equality constraints) ``M``. The dimension of
  box constraints coincides with the optmisation variable dimensionality.

- ``Scalar Type [Optional, Default: double]``: single (``float``) or double precision floating point types; ``complex`` and user-defined types are allowed (compatible with Eigen algebraic kernel)
  but not tested.

- ``Matrix Format [Optional, Default: DENSE]``: possible values: ``DENSE`` and ``SPARSE``. Since dense and sparse matrices in Eigen have slightly different interface, this option
  controls compilation of a specific implementation and optimisations.

- ``Linear System Solver [Optional, Default: Eigen::LDLT (DENSE) | Eigen::SimplicialLDLT (SPARSE)]``: QP solvers implemented in *PolyMPC* support direct and iterative, dense and sparse solvers available in *Eigen*. The user
  can as well provide a custom linear solver given it is derived from the *Eigen* base solver classes and has the same interface.

All QP solvers in *PolyMPC* are derived from the :class:`QPBase` class:

.. code-block:: c++

   template<int N, int M, typename Scalar = double, int MatrixType = DENSE,
           template <typename, int, typename... Args> class LinearSolver = linear_solver_traits<DENSE>::default_solver,
           int LinearSolver_UpLo = Eigen::Lower, typename ...Args>
   class ADMM : public QPBase<ADMM<N, M, Scalar, MatrixType, LinearSolver, LinearSolver_UpLo>, N, M, Scalar, MatrixType, LinearSolver, LinearSolver_UpLo>
   {
   ...
   };

This fairly cumbersome construction allows passing (any) additional arguments (through parameter pack :class:`Args`) that a linear solver potentially might require. It further allows
creating *aliases* for interface types:

- :cpp:class:`qp_var_t`: optimisation vector type (static ``Nx1`` vector)
- :cpp:class:`qp_dual_t`: dual variable (Lagrange multipliers) (static ``(N+M)x1`` vector); access: ``(0..M)``- multipliers associated with general constraints, ``(M...M+N)``- multipliers
  associated with box constraints
- :cpp:class:`qp_hessian_t`: Hessian :math:`P` of the cost function; dense or sparse ``NxN`` matrix

.. NOTE::
   for small and moderate size problems :class:`qp_hessian_t` and other matrices created internally will be static size matrices, for larger
   problems they can go on the heap (dynamic size matrix). This behaviour can be controlled by the compiler definition ``EIGEN_STACK_ALLOCATION_LIMIT``
   Normally, this only affects internal behaviour.

- :cpp:class:`qp_contraint_t`: constraints Jacobian :math:`A`; dense or sparse ``MxN`` matrix
- :cpp:class:`qp_dual_a_t`: dual variable associated with general constraints (static ``Mx1`` vector)

.. function:: status_t QPBase::solve

The class :class:`QPBase` provides purely virtual :func:`solve` which is a placeholder for the user-defined implementation.

.. code-block:: c++

   status_t solve(const Eigen::Ref<const qp_hessian_t>&H, const Eigen::Ref<const qp_var_t>& h,
                  const Eigen::Ref<const qp_constraint_t>& A,
                  const Eigen::Ref<const qp_dual_a_t>& Alb, const Eigen::Ref<const qp_dual_a_t>& Aub,
                  const Eigen::Ref<const qp_var_t>& xlb, const Eigen::Ref<const qp_var_t>& xub) noexcept
   {
       return static_cast<Derived*>(this)->solve_impl(H, h, A, Alb, Aub, xlb, xub);
   }

This assumes to solve the folowing problem:

.. math::
   \begin{equation}
   \begin{array}{ll}
   \underset{x}{\text{minimize}}  & \frac{1}{2} x^T H x + h^T x \\
   \mbox{subject to} & Alb \leq A x \leq Aub \\
   & xlb \leq x \leq xub
   \end{array}
   \end{equation}


Examples
========

Let us now consider several examples that demonstrate the interface of the solver. Assume, we need to solve a simple QP.

.. math::
   \begin{equation}
   \begin{split}
   &\min_{x} \; \frac{1}{2} x^T \begin{bmatrix}
   4 & 1 \\
   1 & 2 \\
   \end{bmatrix} x + \begin{bmatrix} 1 \\ 1  \end{bmatrix}^T x \\
   &\begin{split}
   s.t. \quad & 1 \leq x_1 + x_2 \leq 1 \\
   & \begin{bmatrix} 0 \\ 0 \end{bmatrix} \leq x \leq \begin{bmatrix} 0.7 \\ 0.7 \end{bmatrix}
   \end{split}
   \end{split}
   \end{equation}

**Simple QP: basic ADMM solver**

To solve this problem with *PolyMPC* one might to write a simple program:

.. code-block:: c++

   #include "solvers/admm.hpp"

   int main(void){

   using Scalar = double;

   Eigen::Matrix<Scalar, 2,2> H;
   Eigen::Matrix<Scalar, 2,1> h;
   Eigen::Matrix<Scalar, 1,2> A;
   Eigen::Matrix<Scalar, 1,1> al, au;
   Eigen::Matrix<Scalar, 2,1> xl, xu, solution;

   H << 4, 1,
        1, 2;
   h << 1, 1;
   A << 1, 1;
   al << 1; xl << 0, 0;
   au << 1; xu << 0.7, 0.7;
   solution << 0.3, 0.7;

   const int N = 2; // number of optimisation variables
   const int M = 1; // number of generic constraints

   /** here further template arguments are omitted, and default values are used: dense matrices and LDLT linear solver */
   ADMM<N, M, Scalar> solver;

   solver.solve(H, h, A, al, au, xl, xu);
   Eigen::Vector2d sol = solver.primal_solution();
   }


**Simple QP: boxADMM**

:class:`boxADMM` solver can be created in the similar manner:

.. code-block:: c++

   #include "solvers/box_admm.hpp"

   ...

   const int N = 2; // number of optimisation variables
   const int M = 1; // number of generic constraints

   /** here further template arguments are omitted, and default values are used: dense matrices and LDLT linear solver */
   boxADMM<N, M, Scalar> solver;

   /** the user now can solve any positive semi-definite QP that has N variables and M constraints */
   solver.solve(H, h, A, al, au, xl, xu);
   Eigen::Vector2d sol = solver.primal_solution();


**SimpleQP: Iterative Linear Solver**

Now let's for sake of example pretend that we need an iterative solver, for instance Conjugate Gradient method, to solve
our problem. Moreover, we decide that the problem is nicely scaled (or we do not have enough memory on our chip) and single
precision arithmetics is enough for our purpose and the Hessian is symmetric. Generally, iterative solvers should be used for large sparse problems
(preferably well conditioned) where performing a direct factorisation is expensive.

.. code-block:: c++

   #include "solvers/box_admm.hpp"

   int main(void)
   {

   using Scalar = float;

   Eigen::Matrix<Scalar, 2,2> H;
   Eigen::Matrix<Scalar, 2,1> h;
   Eigen::Matrix<Scalar, 1,2> A;
   Eigen::Matrix<Scalar, 1,1> al, au;
   Eigen::Matrix<Scalar, 2,1> xl, xu, solution;

   H << 4, 1,
        1, 2;
   h << 1, 1;
   A << 1, 1;
   al << 1; xl << 0, 0;
   au << 1; xu << 0.7, 0.7;
   solution << 0.3, 0.7;

   const int N = 2; // number of optimisation variables
   const int M = 1; // number of generic constraints

   /** tell boxADMM to use ConjugateGradient solver available in Eigen (Eigen::ConjugateGradient)
   and tell it the H matrix is symmetric (Eigen::Lower | Eigen::Upper) */
   boxADMM<N, M, Scalar, DENSE, Eigen::ConjugateGradient, Eigen::Lower | Eigen::Upper> solver;

   solver.solve(H, h, A, al, au, xl, xu);
   Eigen::Vector2f sol = solver.primal_solution();
   }

.. NOTE::
   In all previous examples :math:`H` and :math:`A` matrices are defined as static. It is possible, however,
   to provide dynamically allocated matrices, i.e. :class:`MatrixXd` for example. The user should make sure that
   the memory is properly allocated as *PolyMPC* does not perform any data consistency checks in ``RELEASE`` mode.


**SimpleQP: Sparse Matrices**

The problem we are considering is dense. Let's again for the sake of demonstration pretend that the data
in our problem is sparse.

.. code-block:: c++

   #include "solvers/box_admm.hpp"

   int main(void){

   using Scalar = double;

   Eigen::SparseMatrix<Scalar> H(2,2);
   Eigen::SparseMatrix<Scalar> A(1,2);
   Eigen::Matrix<Scalar, 2,1> h;
   Eigen::Matrix<Scalar, 1,1> al, au;
   Eigen::Matrix<Scalar, 2,1> xl, xu, solution;

   /** reserve memory and fill-in the matrices */
   H.reserve(2); H.insert(0,0) = 4; H.insert(0,1) = 1; H.insert(1,0) = 1; H.insert(1,1) = 2;
   A.reserve(1); A.insert(0,0) = 1; A.insert(0,1) = 1;
   h << 1, 1;
   al << 1; xl << 0, 0;
   au << 1; xu << 0.7, 0.7;
   solution << 0.3, 0.7;

   const int N = 2; // number of optimisation variables
   const int M = 1; // number of generic constraints

   /** tell boxADMM to use sparse linear algebra and (default) SimplicialLDLT method */
   boxADMM<N, M, Scalar, SPARSE, linear_solver_traits<SPARSE>::default_solver> solver;

   solver.solve(H, h, A, al, au, xl, xu);
   Eigen::Vector2d sol = solver.primal_solution();
   }


.. NOTE::
   PolyMPC does not assume any particular structure of the sparse matrices. It is responsibility of the user to fill-in data correctly, the
   sparsity pattern will inferred automatically, therefore matrices are assumed to be in **uncompressed** form (default in Eigen). If the sparsity
   patter of the problem does not change in-between solves (only entries values rather), it is possible to set :class:`solver.settings().reuse_pattern = true;`
   which will skip the memory check and allocation step. This feature significantly improves the performance. A full set of avaliable options is available in
   the Settings section.


**SimpleQP: Sparse Iterative Solver**

The iterative solvers can be called the same way with the sparse QP solvers. (For the problem setup see previous example).

.. code-block:: c++

   #include "solvers/admm.hpp"

   ...

   ADMM<M, N, scalar, SPARSE, Eigen::ConjugateGradient, Eigen::Lower | Eigen::Upper> solver;
   /** or: boxADMM<M, N, Scalar, SPARSE, Eigen::ConjugateGradient, Eigen::Lower | Eigen::Upper> solver; */
   solver.solve(H,h,A,Al,Au,xl,xu);
   Eigen::Vector2d sol = solver.primal_solution();


**SimpleQP: OSQP Interface**

To use *OSQP* interface make sure that *OSQP* itself and *osqp-eigen* are installed. You will also need to link your executable to the
:class:`OsqpEigen::OsqpEigen` target.

:class:`osqp_test.cpp`:

.. code-block:: c++

   #include "solvers/osqp_interface.hpp"
   int main(void){

   using Scalar = double;

   Eigen::SparseMatrix<Scalar> H(2,2);
   Eigen::SparseMatrix<Scalar> A(1,2);
   Eigen::Matrix<Scalar, 2,1> h;
   Eigen::Matrix<Scalar, 1,1> al, au;
   Eigen::Matrix<Scalar, 2,1> xl, xu, solution;

   /** reserve memory and fill-in the matrices */
   H.reserve(2); H.insert(0,0) = 4; H.insert(0,1) = 1; H.insert(1,0) = 1; H.insert(1,1) = 2;
   A.reserve(1); A.insert(0,0) = 1; A.insert(0,1) = 1;
   h << 1, 1;
   al << 1; xl << 0, 0;
   au << 1; xu << 0.7, 0.7;
   solution << 0.3, 0.7;

   const int N = 2; // number of optimisation variables
   const int M = 1; // number of generic constraints

   polympc::OSQP<N, M, Scalar> solver;

   solver.solve(H, h, A, al, au, xl, xu);
   Eigen::Vector2d sol = solver.primal_solution();
   }

In your :class:`CMakeLists.txt`:

.. code-block:: bash

   find_package(OsqpEigen)

   add_executable(osqp_test osqp_test.cpp)
   target_link_libraries(osqp_test OsqpEigen::OsqpEigen)

**SimpleQP: QPMAD Interface**

Make sure that *QPMAD* is installed.

.. code-block:: c++

   #include "solvers/qpmad_interface.hpp"

   int main(void){

   using Scalar = double;

   Eigen::Matrix<Scalar, 2,2> H;
   Eigen::Matrix<Scalar, 2,1> h;
   Eigen::Matrix<Scalar, 1,2> A;
   Eigen::Matrix<Scalar, 1,1> al, au;
   Eigen::Matrix<Scalar, 2,1> xl, xu, solution;

   H << 4, 1,
        1, 2;
   h << 1, 1;
   A << 1, 1;
   al << 1; xl << 0, 0;
   au << 1; xu << 0.7, 0.7;
   solution << 0.3, 0.7;

   const int N = 2; // number of optimisation variables
   const int M = 1; // number of generic constraints

   /** here further template arguments are omitted, and default values are used: dense matrices and LDLT linear solver */
   polympc::QPMAD<N, M, Scalar> solver;

   solver.solve(H, h, A, al, au, xl, xu);
   Eigen::Vector2d sol = solver.primal_solution();
   }

.. NOTE::
   *QPMAD* and *OSQP* interfaces accept only problem dimensions and scalar type as template parameters.


Solver Settings
===============

.. function::  const QPBase::settings_t& settings() const noexcept
.. function::  QPBase::settings_t& settings() noexcept

Settings of all solvers can be accessed for writing and reading using :func:`settings()` function which returns
the structure ``settings_t = qp_solver_settings_t<scalar_t>`` containing settings for **all** available solvers. Settings for each QP solver are summarised in the
table below.

**Common Settings**

+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| Setting                        | Description                                                 | Allowed values                                               | Default value   |
+================================+=============================================================+==============================================================+=================+
| :code:`max_iter`               | Maximum number of iterations                                | 0 < :code:`max_iter` (integer)                               | 1000            |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`eps_abs`                | Absolute tolerance                                          | 0 <= :code:`eps_abs`                                         | 1e-03           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`eps_rel`                | Relative tolerance                                          | 0 <= :code:`eps_rel`                                         | 1e-03           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`verbose`                | Verbose output                                              | true/false                                                   | false           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`warm_start`             | Warm start solver                                           | true/false                                                   | false           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`reuse_pattern`          | Skip sparsity estimation and memory allocation step         | true/false                                                   | false           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+

**Settings for ADMM-based Solvers**

+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| Setting                        | Description                                                 | Allowed values                                               | Default value   |
+================================+=============================================================+==============================================================+=================+
| :code:`rho`                    | ADMM rho step                                               | 0 < :code:`rho`                                              | 0.1             |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`adaptive_rho`           | Adaptive rho                                                | true/false                                                   | true            |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`adaptive_rho_interval`  | Adapt rho every N-th iteration                              | 0 (automatic) or 0 < :code:`adaptive_rho_interval` (integer) | 25              |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`adaptive_rho_tolerance` | Tolerance for adapting rho                                  | 1 <= :code:`adaptive_rho_tolerance`                          | 5               |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`sigma`                  | ADMM sigma step                                             | 0 < :code:`sigma`                                            | 1e-06           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`alpha`                  | ADMM overrelaxation parameter                               | 0 < :code:`alpha` < 2                                        | 1.0             |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`check_termination`      | Check termination after N-th iteration                      | 0 (disabled) or 0 < :code:`check_termination` (integer)      | 25              |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+

**Settings Specific to OSQP**

+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| Setting                        | Description                                                 | Allowed values                                               | Default value   |
+================================+=============================================================+==============================================================+=================+
| :code:`eps_prim_inf`           | Primal infeasibility tolerance                              | 0 <= :code:`eps_prim_inf`                                    | 1e-04           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`eps_dual_inf`           | Dual infeasibility tolerance                                | 0 <= :code:`eps_dual_inf`                                    | 1e-04           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`delta`                  | Polishing regularization parameter                          | 0 < :code:`delta`                                            | 1e-06           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`polish`                 | Perform polishing                                           | true/false                                                   | false           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`scaling`                | Number of scaling iterations                                | 0 (disabled) or 0 < :code:`scaling` (integer)                | 10              |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`polish_refine_iter`     | Refinement iterations in polish                             | 0 < :code:`polish_refine_iter` (integer)                     | 3               |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`scaled_termination`     | Scaled termination conditions                               | True/False                                                   | False           |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`adaptive_rho_fraction`  | Adaptive rho interval as fraction of setup time (auto mode) | 0 < :code:`adaptive_rho_fraction`                            | 0.4             |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`time_limit`             | Run time limit in seconds                                   | 0 (disabled) or 0 <= :code:`time_limit`                      | 0               |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| :code:`osqp_linear_solver`     | Linear systems solver type                                  | 0 = LDLT, 1 = Pardiso                                        | LDLT            |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+

**Settings Specific to QPMAD**

+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+
| Setting                        | Description                                                 | Allowed values                                               | Default value   |
+================================+=============================================================+==============================================================+=================+
| :code:`hessian_type`           | Hessian structure                                           | 0:undefined, 1: lower_triangular, 2: cholesky_factor (int)   | 1               |
+--------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+-----------------+


Solver Status
=============

The software provides a unified status for all solvers defined in the :class:`qp_status_t` enum.

+----------------------------+
| status_t                   |
+============================+
| SOLVED                     |
+----------------------------+
| MAX_ITER_EXCEEDED          |
+----------------------------+
| UNSOLVED                   |
+----------------------------+
| UNINITIALIZED              |
+----------------------------+
| INCONSISTENT               |
+----------------------------+

Solver Info
===========

Some statistics of the QP solver including *number of iterations*, *primal and dual residuals*, *status* can accessed with the :func:`info()` function.









