#!/bin/bash -ex

for BETA in 0.00001 0.0001 0.001
do
  for lr in 0.00001 0.00005 0.0001 0.0005
  do
    for seed in 1 2 3 4 5
    do
      # CLUB Model
      # python CLUB_MNIST_IB.py --model CLUB --epochs 200 --seed $seed --lr $lr --BETA $BETA
      
      # vCLUB Model
      python CLUB_MNIST_IB.py --model vCLUB --epochs 200 --seed $seed --lr $lr --BETA $BETA
    done
  done
done