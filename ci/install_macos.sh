#!/bin/sh

set -e

# CasADi
wget -qO casadi.tar.gz "https://github.com/casadi/casadi/releases/download/3.0.0-rc2/osx__casadi-matlabR2015a-v3.0.0-rc2.tar.gz"
mkdir -p ~/casadi
tar xzf casadi.tar.gz -C ~/casadi

# googletest
wget https://github.com/google/googletest/archive/release-1.8.1.tar.gz
tar xzf release-1.8.1.tar.gz
cd googletest-release-1.8.1
mkdir -p build && cd build
cmake ..
make
make install

# Eigen
# wget http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz
# mkdir ~/eigen
# tar xzf 3.3.7.tar.gz -C ~/eigen --strip-components=1
# cd ~/eigen
# mkdir -p build && cd build
# cmake ..
# make install
brew install eigen
