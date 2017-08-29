#!/bin/bash

export MONO_IOMAP=all


die () {
    echo >&2 "$@"
    exit 1
}
[ "$#" -eq 3 ] || die "features_dir and target_dir arguments required, $# provided"


features_dir=$1
target_dir=$2 
layers_list=$3
features_path="/home/ubuntu/efs/features/"$features_dir
result_dir="/home/ubuntu/efs/wf_results/"$target_dir

echo "source dir is : " $features_path
echo "target dir is : " $result_dir
echo "layers list are is : " $layers_list

echo "create directory if not exists"
mkdir -p $result_dir

# calculate alpha for the layer
foo () {
	local layer=$1
	#make dir where data, labels and application are copied
	rm -rf $result_dir/$layer
	mkdir $result_dir/$layer
	
	# copy data and label
	#directory for execution should not  start with t_, tmp_, test_ . (otherwise wf.exe will not work)
	cp $features_path/$layer'.csv' $result_dir/$layer/trainingData.csv
	cp $features_path/trainingLabel.csv $result_dir/$layer/

	# copy application and config
	cp /home/ubuntu/BesovSmoothness/config.txt $result_dir/$layer/
	cp /home/ubuntu/BesovSmoothness/wf.exe $result_dir/$layer/

	# rename file path in config.txt.
	# BE CAREFULL! tabs should be in path!!!
	line1="dbPath			"$result_dir/$layer/
	line2="resultsPath			"$result_dir/"alpha_"$layer/
	sed -i "1s#.*#$line1#" $result_dir/$layer/config.txt
	sed -i "2s#.*#$line2#" $result_dir/$layer/config.txt

	# start wf estimation
	# wf.exe should be executed within directory with data!!!
	echo "start wf for " $layer
	( cd $result_dir/$layer ; mono wf.exe )
	
	echo "alpha for " $layer " is "$(cat $result_dir/alpha_$layer/0/alpha.txt)
	# remove copied layer features
	rm -r $result_dir/$layer
}

echo "start loop"
for layer in $layers_list; do foo $layer & done
wait
