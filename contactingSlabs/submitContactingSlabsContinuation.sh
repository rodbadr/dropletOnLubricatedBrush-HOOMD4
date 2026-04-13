#!/bin/bash

b=1
start=$1
end=$2
ngpu=$3

. ~/.hoomd4Ashbaugh/bin/activate

for i in `seq $start $end`;
do

	c=$(($i+$b))
	echo "TASK $c"
	python3 contactingSlabsContinuation.py $c >log/run$c.out 2>&1 &


	d=$(($c-$start))
	if [ $(($d%$ngpu)) -eq 0 ]
	then
		wait
	fi

done
