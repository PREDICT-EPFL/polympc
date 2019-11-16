#!/bin/sh

set -e

mkdir -p build
cd build
cmake -DBUILD_TESTS=ON -DBUILD_RELEASE=ON ..
make
