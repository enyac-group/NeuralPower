#! /bin/sh
#source ~/tensorflow/bin/activate
NET_FILES=./nets/densenet.json
OUT_FILE=./results/tmp.txt

echo '' > $OUT_FILE
for NET_FILE in $NET_FILES 
  do
  echo '\n\n' >> $OUT_FILE
#  ./paleo.sh fullpass $NET_FILE >> $OUT_FILE

./paleo.sh profile $NET_FILE \
    --direction=forward \
    --executor=tensorflow \
    >> $OUT_FILE
done
    exit
