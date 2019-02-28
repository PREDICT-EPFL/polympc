#!/bin/sh

# Make the script fail if any command in it fails
set -e

# Install dependencies
apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    wget
apt-get clean

# Get CasADi
wget -qO libcasadi.deb "https://github.com/casadi/casadi/releases/download/3.0.0-rc2/linux__libcasadi-v3.0.0-rc2.deb"
dpkg -i libcasadi.deb
apt-get install -f
rm libcasadi.deb
