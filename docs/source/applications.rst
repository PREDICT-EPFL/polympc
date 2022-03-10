.. _chapter-app:

=========================
Examples and Applications
=========================

This chapter contains some examples and robotic applications using *PolyMPC*.

Thrust Vector Control of a Rocket
=================================

Thrust Vector Control (TVC) is a key technology enabling rockets to perform complex autonomous missions, such as active stabilization, orbit insertion, or propulsive landing.
This is achieved by independently controlling the thrust direction and magnitude of each of its engines. Compared to aerodynamic control such as fins or canard, it guarantees
a high control authority even in the absence of an atmosphere, i.e. during high altitude launches or exploration of other planetary bodies.

.. image:: img/drone.jpg

.. image:: img/drone_gimbal.jpg

Dynamics:

.. math::

    \begin{equation}
    \begin{split}
        &\dot{X}
        =
        f(X, U)
        =
        \begin{bmatrix}
        \dot{p}\\
        \dot{v}\\
        \dot{q}\\
        \dot{\omega^b}\\
        \end{bmatrix}
        =
        \begin{bmatrix}
        v\\
        \cfrac{R(q) F^b_T}{m} + g\\
        \frac{1}{2} q \circ \omega^b\\
        I^{-1} (M^b - \omega^b \times (I\omega^b))\\
        \end{bmatrix}\\
        &M^b = r \times F^b_T + M^b_P\\
    \end{split}
    \end{equation}


with :math:`I` the inertia matrix of the drone, and :math:`r` the position of the thrust :math:`F^b_T` from the center of mass.

OCP:

.. math::

    \begin{equation}
    \begin{split}
    &\min_{u(t)} \; \int_{t_0}^{t_f} l(x, u, t) dt + V_f(x_f)\\
        &\begin{split}
        s.t. \quad & \dot{x} = f(x, u)\\
        & -\theta_{max} \leq \theta_1 \leq \theta_{max}\\
        & -\theta_{max} \leq \theta_2 \leq \theta_{max}\\
        & -\dot{\theta}_{max} \leq \dot{\theta_1} \leq \dot{\theta}_{max}\\
        & -\dot{\theta}_{max} \leq \dot{\theta_2} \leq \dot{\theta}_{max}\\
        & P_{min} \leq \bar{P} + {P_\delta}/2 \leq P_{max}\\
        & P_{min} \leq \bar{P} - {P_\delta}/2 \leq P_{max}\\
        & {P_\delta}_{min} \leq P_\delta \leq {P_\delta}_{max}\\
        & 0 \leq z\\
        \end{split}
    \end{split}
    \end{equation}


**TODO...**
