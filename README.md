<h1 align="center">Pytorch Keypoint Detection</h1>

This repo contains a Python package for 2D keypoint detection using [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) and [wandb](https://docs.wandb.ai/). Keypoints are trained using Gaussian Heatmaps, as in [Jakab et Al.](https://proceedings.neurips.cc/paper/2018/hash/1f36c15d6a3d18d52e8d493bc8187cb9-Abstract.html) or [Centernet](https://github.com/xingyizhou/CenterNet).

This package is been used for research at the [AI and Robotics](https://airo.ugent.be/projects/computervision/) research group at Ghent University. You can see some applications below: The first image shows how this package is used to detect corners of cardboard boxes, in order to close the box with a robot. The second example shows how it is used to detect a varying number of flowers.
<div align="center">
  <img src="doc/img/box-keypoints-example.png" width="80%">
  <img src="doc/img/keypoints-flowers-example.png" width="80%">
</div>


## Main Features

- This package contains **different backbones** (Unet-like, dilated CNN, Unet-like with pretrained ConvNeXt encoder). Furthermore you can  easily add new backbones or loss functions. The head of the keypoint detector is a single CNN layer.
- The package uses the often-used **COCO dataset format**.
- The detector can deal with an **arbitrary number of keypoint channels**, that can contain **a varying amount of keypoints**. You can easily configure which keypoint types from the COCO dataset should be mapped onto the different channels of the keypoint detector.
- The package contains an implementation of the Average Precision metric for keypoint detection.
- Extensive **logging to wandb is provided**: The loss for each channel is logged, together with the AP metrics for all specified treshold distances. Furthermore, the raw heatmaps, detected keypoints and ground truth heatmaps are logged at every epoch for the first batch to provide insight in the training dynamics and to verify all data processing is as desired.
- All **hyperparameters are configurable** using a python argumentparser or wandb sweeps.

note: this is the second version of the package, for the older version that used a custom dataset format, see the github releases.


TODO: add integration example.

## Local Installation
- clone this repo in your project (e.g. as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules), using [vcs](https://github.com/dirk-thomas/vcstool),..). It is recommended to lock to the current commit as there are no guarantees w.r.t. backwards comptability.
- create a conda environment using `conda env create --file environment.yaml`
- activate with `conda activate keypoint-detection`
- run `wandb login` to set up your wandb account.
- you are now ready to start training.

## Dataset

This package used the [COCO format](https://cocodataset.org/#format-data) for keypoint annotation and expects a dataset with the following structure:
```
dataset/
  images/
    ...
  <name>.json : a COCO-formatted keypoint annotation file.
```
For an example, see the `test_dataset` at `test/test_dataset`.


### Labeling
If you want to label data, we provide integration with the [CVAT](https://github.com/opencv/cvat) labeling tool: You can annotate your data and export it in their custom format, which can then be converted to COCO format. Take a look [here](labeling/Readme.md) for more information on this workflow and an example. To visualize a given dataset, you can use the  `keypoint_detection/utils/visualization.py` script.

## Training

There are 2 ways to train the keypoint detector:

- The first is to run the `train.py` script with the appropriate arguments. e.g. from the root folder of this repo, you can run the bash script `bash test/integration_test.sh` to test on the provided test dataset, which contains 4 images. You should see the loss going down consistently until the detector has completely overfit the train set and the loss is around the entropy of the ground truth heatmaps (if you selected the default BCE loss).

- The second method is to create a sweep on [wandb](https://wandb.ai) and to then start a wandb agent from the correct relative location.
A minimal sweep example  is given in `test/configuration.py`. The same content should be written to a yaml file according to the wandb format. The sweep can be started by running `wandb agent <sweep-id>` from your CLI.


To create your own configuration: run `python train.py -h` to see all parameter options and their documentation.

## Using a trained model (Inference)
During training Pytorch Lightning will have saved checkpoints. See `scripts/inference.py` for a simple example to run inference with a checkpoint.

## Development  info
- formatting and linting is done using [pre-commit](https://pre-commit.com/)
- testing is done using pytest (with github actions for CI)


## Note on performance
- Keep in mind that the Average Precision is a very expensive operation, it can easily take as long to calculate the AP of a .1 data split as it takes to train on the remaining 90% of the data. Therefore it makes sense to use the metric sparsely. The AP will always be calculated at the final epoch, so for optimal train performance (w/o intermediate feedback), you can e.g. set the `ap_epoch_start` parameter to your max number of epochs + 1.

## Rationale:
TODO
- why this repo?
  - why not label keypoints as bboxes and use YOLO/Detectron2?
  - ..
