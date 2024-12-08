#!/bin/bash


for i in $(seq -w 0 63); do
  job_name="VSRD_TEST_SPLIT${i}"
  config_path=$(printf "%02d" $i)
  
  temp_script="run_temp_${i}.sh"
  
  cp tsubame_ddp_example.sh $temp_script
  
  sed -i "s/#\$ -N VSRD_TEST_SPLIT01/#\$ -N ${job_name}/" $temp_script
  sed -i "s/--config_path \"00\"/--config_path \"${config_path}\"/" $temp_script
  
  echo "Generated Files Name: $temp_script"
done

