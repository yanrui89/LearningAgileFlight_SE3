#!/bin/bash

for i in 50
do
   for alpha in 10 100
   do
      for beta in 10 100
      do

         python deep_learning.py $i $alpha $beta 10
      done
   done
done