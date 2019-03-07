#!/bin/sh

set -e

wget -qO casadi.tar.gz "https://github.com/casadi/casadi/releases/download/3.0.0-rc2/osx__casadi-matlabR2015a-v3.0.0-rc2.tar.gz"
mkdir -p ~/casadi
tar xzf casadi.tar.gz -C ~/casadi
