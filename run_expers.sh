#!/bin/bash

## Usage
## bash run_expers.sh > commands
## cat commands | xargs -n1 -P8 -I{} /bin/sh -c "{}" 

# some buffer for GPU scheduling at the start
echo sleep 20
# echo sleep 40
# echo sleep 60
# echo sleep 80
# echo sleep 100
# echo sleep 120
# echo sleep 140

# for i in 1 2 3 4 5
for i in 1
do

    for probType in simple
    do
        echo python method.py --probType $probType
        echo python baseline_opt.py --probType $probType
        echo python method_deeplde.py --probType $probType
        echo python method_pdl.py --probType $probType
        echo python method_gauge.py --probType $probType

        numIneq=100
        numEq=100
        numVar=200

        echo python method.py --probType $probType --simpleVar $numVar --simpleIneq $numIneq --simpleEq $numEq
        echo python baseline_opt.py --probType $probType --simpleVar $numVar --simpleIneq $numIneq --simpleEq $numEq
        echo python method_deeplde.py --probType $probType --simpleVar $numVar --simpleIneq $numIneq --simpleEq $numEq
        echo python method_pdl.py --probType $probType --simpleVar $numVar --simpleIneq $numIneq --simpleEq $numEq
        echo python method_gauge.py --probType $probType --simpleVar $numVar --simpleIneq $numIneq --simpleEq $numEq
    done

    for probType in nonconvex
    do
        echo python method.py --probType $probType
        echo python baseline_opt.py --probType $probType
        echo python method_deeplde.py --probType $probType
        echo python method_pdl.py --probType $probType
        echo python method_gauge.py --probType $probType

        numIneq=100
        numEq=100
        numVar=200

        echo python method.py --probType $probType --nonconvexVar $numVar --nonconvexIneq $numIneq --nonconvexEq $numEq
        echo python baseline_opt.py --probType $probType --nonconvexVar $numVar --nonconvexIneq $numIneq --nonconvexEq $numEq
        echo python method_deeplde.py --probType $probType --nonconvexVar $numVar --nonconvexIneq $numIneq --nonconvexEq $numEq
        echo python method_pdl.py --probType $probType --nonconvexVar $numVar --nonconvexIneq $numIneq --nonconvexEq $numEq
        echo python method_gauge.py --probType $probType --nonconvexVar $numVar --nonconvexIneq $numIneq --nonconvexEq $numEq
    done

    # for probType in simple
    # do
    #     numIneq=100
    #     numEq=100
    #     numVar=200

    #     echo python method.py --probType $probType --simpleVar $numVar --simpleIneq $numIneq --simpleEq $numEq
    #     echo python baseline_opt.py --probType $probType --simpleVar $numVar --simpleIneq $numIneq --simpleEq $numEq
    #     echo python method_deeplde.py --probType $probType --simpleVar $numVar --simpleIneq $numIneq --simpleEq $numEq --hiddenSize 512 --prefix "/data1/jxxiong/DC2/"
    #     echo python method_pdl.py --probType $probType --simpleVar $numVar --simpleIneq $numIneq --simpleEq $numEq --hiddenSize 512 --prefix "/data1/jxxiong/DC2/"
    #     echo python method_gauge.py --probType $probType --simpleVar $numVar --simpleIneq $numIneq --simpleEq $numEq --hiddenSize 512 --prefix "/data1/jxxiong/DC2/"
    # done

    # for probType in nonconvex
    # do
    #     numIneq=100
    #     numEq=100
    #     numVar=200

    #     echo python method.py --probType $probType --nonconvexVar $numVar --nonconvexIneq $numIneq --nonconvexEq $numEq
    #     echo python baseline_opt.py --probType $probType --nonconvexVar $numVar --nonconvexIneq $numIneq --nonconvexEq $numEq
    #     echo python method_deeplde.py --probType $probType --nonconvexVar $numVar --nonconvexIneq $numIneq --nonconvexEq $numEq --hiddenSize 512 --prefix "/data1/jxxiong/DC2/"
    #     echo python method_pdl.py --probType $probType --nonconvexVar $numVar --nonconvexIneq $numIneq --nonconvexEq $numEq --hiddenSize 512 --prefix "/data1/jxxiong/DC2/"
    #     echo python method_gauge.py --probType $probType --nonconvexVar $numVar --nonconvexIneq $numIneq --nonconvexEq $numEq --hiddenSize 512 --prefix "/data1/jxxiong/DC2/"
    # done
done
