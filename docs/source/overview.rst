.. _chapter-overview:

========
Overview
========

* **Performance** - The software is designed to create real-time optimisation
  and predictive control algorithms for fast mechatronic systems that potentially
  can run on low-power computation platforms.  Therefore, PolyMPC avoids unnecessary
  memory allocations, copying and expensive calls of virtual methods while exploiting 
  vectorisation capabilities of modern processors. To allow for embedded deployment on
  microcontrollers without an operating system the software supports fully static 
  memory allocation. The tool supports both dense and sparse linear algebra computations.

* **Usability** - Efficient implementation of state-of-the-art methods while preserving simplicity and
  a user-friendly interface. The use of modern frameworks for linear algebra
  and automatic differentiation allows the user to formulate problems using
  intuitive vector notation.

* **Modularity** - Each algorithmic component of PolyMPC is usable independently
  and interaction of the components happens at zero computational cost.
  Moreover, interfaces and implementations are logically separated, i.e.
  for each family of algorithms (QP, NLP solvers, OCP discretisation etc) we provide
  unifying interfaces.

* **Extensibility** - The tool is designed in a manner that allows the user to
  formulate OCPs as well as easily utilize and modify the building blocks of the
  algorithm by changing or adding a corresponding implementation. 

Problem Classes
===============

* **Quadratic Programming**: dense and sparse quadratic problems; possibility for the full static memory allocation; embedded-friendly numerical algorithms.

* **Nonlinear Programming**: constrained nonlinear problems; dense and sparse computations; automatic sensitivity generation; custom solvers and interfaces to the proven nonlinear codes.

* **Optimal Control Problems**: continuous-time nonlinear OCP; highly efficient implementation of the pseudo spectral collocation methods; automatic transcription and sensitivity generation; possibility for full static allocation.

* **Polynomial computations**: polynomial interpolation and approximation, Gauss quadratures, polynomial chaos expansions (PCE).