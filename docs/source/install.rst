.. _chapter-installation:

============
Installation
============

Getting the source code
=======================

For the latest version, clone the git repository

.. code-block:: bash

    git clone https://github.com/LA-EPFL/polympc.git

.. _section-dependencies:

Dependencies
============

- (**required**) Building tools: `CMake <https://cmake.org/>`_ 3.0 or higher, and a *C++11*-compliant compiler. 

  .. NOTE :: 

     PolyMPC has been tested with *gcc*, *clang*, *qcc*, *Visual Studio* and *Min-GW*. 

- (**required**) PolyMPC primarily relies on `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_ 3.3 or later for linear algebra routines and linear system solvers (dense and sparse).

  .. NOTE ::
     
     On Ubuntu 18.04 and earlier we recommend to install Eigen from source.

- (**optional**) The early versions of the toolbox were implemented using the `CasADi <https://web.casadi.org/>`_ 3.2 (or higher) framework. Currently the CasADi related code is not actively supported or developed.

- (**optional**) `gtest <https://github.com/google/googletest>`_ is used to build tests. See :ref:`section-building` for details.


.. _section-building:

Building the code
=================

Since PolyMPC is a header-based library, there is nothing to be compiled in general. We can, however, build tests and install the solvers as shown below:

.. code-block:: bash

 mkdir build
 cd build
 cmake ../polympc
 # There is nothing build by default
 make 
 # Optionally install PolyMPC (coming)
 make install

*Build configuration*

The following optional CMake flags can be added to customise the build. PolyMPC features unit and integral tests for most of the software functionality. Templates make the compilation process long and undesirable if the tool is used as a third-party library. Therefore, every test set can be compiled separately. All tests are disabled by default.

- ``BUILD_TESTS [Default: OFF]``: Build all tests in PolyMPC (control, QP and NLP solvers, polynomials, autodiff). By default, no tests are compiled.

- ``BUILD_RELEASE [Default: ON]``: Build with optimisations and without debugging information. Enabled by default.

- ``CONTROL_TESTS [Default: OFF]``: Build control test: optimal control problems, MPC, LQR.

- ``QP_TESTS [Default: OFF]``: Build QP tests: ADMM, boxADMM, OSQP, QPMAD solvers.

- ``SQP_TESTS [Default: OFF]``: Build SQP solver tests.

- ``POLY_TESTS [Default: OFF]``: Build tests for the polynomial module: interpolation, projection, integration.

- ``AUTODIFF_TESTS [Default: OFF]``: PolyMPC features some extensions to the Eigen AD module: splines, special function and some more. 

- ``BUILD_EXAMPLES [Default: OFF]``: Build some examples (CasADi and Eigen): kite control, CSTR reactor, car control and some more.






