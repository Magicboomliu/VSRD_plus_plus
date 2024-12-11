DIRECTLY_INFERENCE(){

export PYTHONPATH="${PYTHONPATH}:/home/zliu/TPAMI25/AutoLabels/SDFlabel/sdfrenderer"
filename="/home/zliu/TPAMI25/AutoLabels/SDFlabel/data_preprocssing/all_filenames.txt"
python Inference.py --filename $filename


}


DIRECTLY_INFERENCE