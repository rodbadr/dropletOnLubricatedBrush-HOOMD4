#!/bin/bash

b=1

start=$1
end=$2
ngpu=$3

for i in `seq $start $end`;
do

	c=$(($i+$b))
	echo "TASK $c"
	python3 liqSlabFullBox.py $c >log/runFull$c.out 2>&1 &

	d=$(($i-$start))
	f=$(($d+$b))

	if [ $(($f%$ngpu)) -eq 0 ]
	then
		wait
	fi

done

for i in `seq $start $end`;
do

	c=$(($i+$b))
	echo "TASK $c"
	python3 liqSlabSecondStep.py $c >log/runSecond$c.out 2>&1 &

	d=$(($i-$start))
	f=$(($d+$b))

	if [ $(($f%$ngpu)) -eq 0 ]
	then
		wait
	fi

done
