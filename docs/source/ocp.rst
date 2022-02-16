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

