#!/bin/bash

for i in 100 10
do
   for alpha in 10
   do
      for beta in 10
      do
         for gamma in 10
         do
            for ep_beta in 0.001 0.0001 0.00001 0.000001
            do
               for ep_bool in 1 0
               do
                  python deep_learning.py $i $alpha $beta $gamma $ep_beta $ep_bool
               done
            done
         done
      done
   done
done