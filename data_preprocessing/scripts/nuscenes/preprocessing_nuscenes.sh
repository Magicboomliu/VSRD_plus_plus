Create_Nuscenes_Infos_STEP1() {

cd ../../..

nuscenes_root_path="/data2/zliu_backup_data/nuScenes/"
can_bus_root_path="/data2/zliu_backup_data/nuScenes/"
out_path="/data2/zliu_backup_data/nuScenes/"
version="v1.0-trainval"
max_sweeps=10
info_prefix="nuscenes"

python data_preprocessing/nuscenes/step1_create_infos.py \
    --root_path $nuscenes_root_path \
    --can_bus_root_path $can_bus_root_path \
    --out_path $out_path \
    --info_prefix $info_prefix \
    --version $version \
    --max_sweeps $max_sweeps
}



Add_Dynamic_Flags_STEP2_Train() {
cd ../../..
info_path="/data2/zliu_backup_data/nuScenes/nuscenes_infos_train.pkl"
dynamic_classes="car,truck,trailer,bus,construction_vehicle"
speed_thresh=1.0

python data_preprocessing/nuscenes/step2_add_dynamic_flags.py \
  --info_path $info_path \
  --speed_thresh $speed_thresh \
  --dynamic_classes $dynamic_classes
}

Add_Dynamic_Flags_STEP2_VAL() {
cd ../../..
info_path="/data2/zliu_backup_data/nuScenes/nuscenes_infos_temporal_val.pkl"
dynamic_classes="car,truck,trailer,bus,construction_vehicle"
speed_thresh=1.0


python data_preprocessing/nuscenes/step2_add_dynamic_flags.py \
  --info_path $info_path \
  --speed_thresh $speed_thresh \
  --dynamic_classes $dynamic_classes
}

#Add_Dynamic_Flags_STEP2_Train
Add_Dynamic_Flags_STEP2_VAL