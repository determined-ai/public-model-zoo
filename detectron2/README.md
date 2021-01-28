# Current Status

This repo is based on the facebook detectron2 repo: https://github.com/facebookresearch/detectron2/blob/v0.1.2/tools/plain_train_net.py. In particular version v0.1.2.

The code was developed on Determined: 12.12. The version right before remove-steps.

Currently, we are able to get the correct metrics results, while training at about 52 images per second. Facebook reports receiving 62 images per second: https://detectron2.readthedocs.io/notes/benchmarks.html, due to limited support of IterableDatasets, which is required to be a first class citizen to continue development. 

# Data
Fetching the data can be found in coco.sh. The data and annotations are needed to run. It takes some time to download and unpack the data.

# Model
You must install the detectron repo. It can take some time to install. A docker container can be used to speed it up however, the type of GPU you are using for training must be the same as the ones being built on. For example, you can not build on your local computer a docker container then use that container on a V100. You will receive C level errors.