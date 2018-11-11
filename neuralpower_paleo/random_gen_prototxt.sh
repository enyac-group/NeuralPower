#!/bin/bash
path=~/tf_profiler/paleo/nets/generator
out_dir=$path/random_prototxt
source /home/deep/caffe/bin/activate
:'
if [ ! -d "$out_dir" ]
  then
  mkdir $out_dir
fi
END=200
dataset=cifar10
for ((i=0;i<END;i++));
  do
  python $path/generate_network.py $dataset $out_dir/random_$i.prototxt
done

dataset=mnist
for ((i=END;i<$((END*2));i++));
  do
  python $path/generate_network.py $dataset $out_dir/random_$i.prototxt
done
'

# generate json file
source ~/tf_profiler/paleo/bin/activate
home_dir=/home/deep/tf_profiler/paleo
input_files=$out_dir/*
output_dir=$home_dir/nets/random_json
if [ ! -d "$output_dir" ]
  then
  mkdir $output_dir
fi

for f in $input_files
  do
  filename=$(basename $f)
  extension=${filename#*.}
  filename=${filename%.*}
  if [ "$extension" == 'prototxt' ]
    then
    python $home_dir/paleo/utils/convert.py ${f} > ${output_dir}/${filename}.json
  fi
done