cp -pv /run/determined/workdir/local/_pytorch_trial.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/_pytorch_trial.py
cp -pv /run/determined/workdir/local/_pytorch_context.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/_pytorch_context.py


export DETECTRON2_DATASETS=/mnt/dtrain-fsx/detectron2

# Uncomment if running without a container:
# pip install cython
# pip install pycocotools
# # # python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# python -m pip install 'git+https://github.com/katport/detectron2_fork.git'

# git clone https://github.com/facebookresearch/detectron2.git

# cd detectron2
# cd datasets
# mkdir coco
# mkdir coco/annotations
# cd coco/annotations

# wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# cd ../../../../

# ./prepare_for_tests.sh
