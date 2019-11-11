#!/bin/bash

# set -e # exit on error

tests=(
    "./build/tests/solvers/qp/qp_solver_test"
    "./build/tests/solvers/sqp/sqp_test"
    "./build/tests/control/nmpc_test"
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
