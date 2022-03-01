#!/bin/bash

generate_normal() {
      number=$1
      echo "Processing $1"
      python render_NYUv2.py $number
}


max_num_processes=5
num_processes=20

echo $max_num_processes
echo $num_processes
echo $len

for ((i=0; i<=1448; i++))
do
      ((j=i%num_processes))
      ((j==0)) && ((i!=0)) && wait
      generate_normal $i &
done
wait