# Model Training

The workspaces folder contains subfolders which all the files related to model training for a particular dataset.

```shell
machine-learning-sandbox/
    workspaces/
        my-mobilenet-model-training/
            annotations/        # for .csv and .record files with annotations
            exported-models/    # exported versions of trained models
            images/             # copy of all images in dataset and .xml files for each one
                train/
                test/
            models/             # contains subfolder for each training job
            pretrained-models/  # contains downloaded pretrained models used as checkpoints for training jobs
        README.md               # this general info file for training
```

## Data Preparation
The data annotation is done with [LabelImg](https://github.com/tzutalin/labelImg) graphical image annotation tool. See the link for info on how to install and use. Precompiled binaries are stored in the labelImg directory found as shown below.

```shell
machine-learning-sandbox/
    add-ons/
        labelImg/
```
### Usage
```shell
# From within machine-learning-sandbox/addons/labelImg
python labelImg.py
# or
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```
