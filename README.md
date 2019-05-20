# PolyMPC

[![Build Status](https://travis-ci.com/LA-EPFL/polympc.svg?branch=experimental)](https://travis-ci.com/LA-EPFL/polympc)

PolyMPC: An Efficient and Extensible Tool for Real-Time Nonlinear Model Predictive Tracking and Path Following for Fast Mechatronic Systems. Developed at the Automatic Control laboratory EPFL as part of the AWESCO project.

## Dependencies

 - CasADi (version v3.2.0 and higher) : https://github.com/casadi
 - Eigen3 : https://eigen.tuxfamily.org/

## Build

PolyMPC is based on the CMake building system. In order to build the package:
```
mkdir -p polympc-build
cd polympc-build
cmake ../path_to_polympc_folder/
make
```

## Description

PolyMPC is an open-source software tool for the pseudospectral collocation-based real-time model predictive control.  The tool relies on the CasADi and Eigen frameworks which makes it easier to integrate the tool into existing projects since there is no need to reimplement existing mathematical models using some framework-specific modeling language. Furthermore, a user is not tied to using the implemented predictive controllers but rather is free to utilize each of the software modules independently. Namely, to obtain the functors to evaluate collocated differential equations, approximated integral together with their derivatives, or simply use available integration routines.

## Contact
plistov@gmail.com
