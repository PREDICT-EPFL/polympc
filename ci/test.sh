#!/bin/bash

# set -e # exit on error

tests=(
    "./build/src/qpsolver/osqp_test"
    "./build/src/qpsolver/sqp_test"
    "./build/src/qpsolver/bfgs_test"
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
