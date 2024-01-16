#!/bin/bash

for i in 100
do
   for alpha in 10 100
   do
      for beta in 10 100
      do
         for gamma in 10 100
         do
            python deep_learning_test.py $i $alpha $beta $gamma
         done
      done
   done
done