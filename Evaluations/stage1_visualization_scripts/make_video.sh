Make_Video(){
cd ..
cd tools/Visualizations

result_folder="/home/zliu/CVPR25_Detection/Submitted_Version/VSRD-V2/Evaluations/tools/Visualizations/vis_mlp_only/"
fps=5

python make_video.py  --result_folder $result_folder \
                        --fps $fps
}

Make_Video