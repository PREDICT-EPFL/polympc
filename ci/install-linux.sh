#!/bin/sh

# Make the script fail if any command in it fails
set -e

# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    wget
sudo apt-get clean

# Get CasADi
wget -qO libcasadi.deb "https://github.com/casadi/casadi/releases/download/3.0.0-rc2/linux__libcasadi-v3.0.0-rc2.deb"
sudo dpkg -i libcasadi.deb
sudo apt-get install -f
rm libcasadi.deb

# Eigen
wget http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz
mkdir ~/eigen
tar xzf 3.3.7.tar.gz -C ~/eigen --strip-components=1
cd ~/eigen
mkdir -p build && cd build
cmake ..
sudo make install

# Build & install googletest
wget https://github.com/google/googletest/archive/release-1.8.1.tar.gz
tar xzf release-1.8.1.tar.gz
cd googletest-release-1.8.1
mkdir -p build && cd build
cmake ..
make
sudo make install
