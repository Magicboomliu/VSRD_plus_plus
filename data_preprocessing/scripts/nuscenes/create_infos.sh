Create_Nuscenes_Infos() {

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

Create_Nuscenes_Infos