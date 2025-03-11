#!/bin/bash

## Usage
## bash run_expers_search.sh > commands_search
## cat commands_search | xargs -n1 -P8 -I{} /bin/sh -c "{}" 

# some buffer for GPU scheduling at the start
echo sleep 20
echo sleep 40
echo sleep 60
echo sleep 80
echo sleep 100
echo sleep 120
echo sleep 140

i=0 
for tau in 0.8 
do
    for rho in 0.1 0.5 10
    do
        for rho_max in 5000 10000
        do
            for alpha in 1.5 2 5 10
            do  
                for hidden in 500 1000
                do
                    ((i++))
                    probType=simple
                    echo "python method_pdl.py --hiddenSize $hidden --simpleVar 200 --simpleIneq 100 --simpleEq 100 --probType $probType --tau $tau --rho $rho --rho_max $rho_max --alpha $alpha > logs/${probType}_100_pdl_${i}.log"
                    # probType=nonconvex
                    # echo "python method_pdl.py --probType $probType --tau $tau --rho $rho --rho_max $rho_max --alpha $alpha > logs/${probType}_50_pdl_${i}.log"
                done
            done
        done
    done
done


# i=0
# for rho in 0.0001 0.0005 0.01 0.05 0.1 0.5
# do 
#     for lambda in 0.0 0.1
#     do 
#         ((i++))
#         probType=simple
#         echo "python method_deeplde.py --probType $probType --rho $rho --lambda $lambda > logs/${probType}_50_deeplde_${i}.log"
#     done
# done

python method.py --probType simple --useTrainCorr False --useTestCorr False --softWeight 50 > logs/simple_50_method.log