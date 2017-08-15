#!/bin/bash

export MONO_IOMAP=all

layers_path="/home/ubuntu/hidden_layers/features/model_mnist_c19"
new_dir_name="res_model_mnist_c19"
result_dir="/home/ubuntu/hidden_layers/wf_results/"$new_dir_name

echo "delete target directory"
rm -rf $result_dir
mkdir $result_dir
echo "start loop"

# these are layers names 

for layer in f6 f0 f5 f4 f2 f3 f1
do
		#make dir where data, labels and application are copied
		mkdir $result_dir/$layer
		
		# copy data and label
		#directory for execution should not  start with t_, tmp_, test_ . (otherwise wf.exe will not work)
		cp $layers_path/$layer'.csv' $result_dir/$layer/trainingData.csv
        cp $layers_path/trainingLabel.csv $result_dir/$layer/

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
		
		# remove copied layer features
		#rm -r $result_dir/$layer
done
