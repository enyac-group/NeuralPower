#!/bin/bash
source ~/tf_profiler/paleo/bin/activate
home_dir=/home/deep/tf_profiler/paleo
input_files=$home_dir/nets/caffe/*
output_dir=$home_dir/nets/known_json
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


