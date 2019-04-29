#!/bin/bash

# set -e # exit on error

tests=(
    "./build/src/solvers/qp_solver_test"
    "./build/src/solvers/sqp_test"
    "./build/src/solvers/sqp_test_autodiff"
    "./build/src/solvers/bfgs_test"
    "./build/src/control/nmpc_test"
)

FAIL=1
PASS=0

# Run all tests before failing
result=$PASS
for t in "${tests[@]}"
do
    echo $t
    $t
    if [ $? == 1 ]; then
        echo "Test \"$t\" failed"
        result=$FAIL
    fi
done

exit $result
