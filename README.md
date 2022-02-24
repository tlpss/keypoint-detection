# keypoint-detection
Python package for 2D keypoint detection using [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) and [wandb](https://docs.wandb.ai/). Keypoints are trained using Gaussian Heatmaps (cf CenterNet paper).

**Main Features**

- This package contains a number of backbones (Unet-like, dilated CNN,...)and loss functions. Both are modular with a factory pattern, which allows to easily add new backbones or loss functions. The head of the keypoint detector is a single CNN layer (as usual).
- The detector can deal with an arbitrary number of keypoint classes (or channels), that can deal with different number of keypoints within a batch.
- The package contains an implementation of the meanAP metric for keypoints detection.
- All hyperparameters are configurable using a python argumentparser or wandb.
- Extensive logging to wandb is provided: The loss for each channel is logged, together with the AP metrics for all specified treshold distances. Furthermore, the raw heatmaps and ground truth heatmaps are logged at every epoch for the first batch to provide insight in the training dynamics and to verify all data processing is as desired.

For an example integration of the package in your own project, see [here](https://github.com/tlpss/box-manipulation).

## Local Installation
- clone this repo in your project (as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules), using [vcs](https://github.com/dirk-thomas/vcstool),..).
- create conda env using `conda env create --file environment.yaml`
- activate the environmant using `conda activate python39`
- pip install the package using `pip install -e keypoint_detection` from the repo root directory. This will install the required dependencies in the current python environment.
- run `wandb login` to set up your wandb account.


## Dataset format

This package expects a dataset with the following format:
- a folder which contains all the images
- a `.json` file that defines the datapoints and has following structure:

```
{
	"dataset": [
	{
      "image_path": "relative-path-to-img.png",
      "channelX": [
        [
          0.55,
          0.35,
          1.02
        ],
        [
          0.34,
          0.46,
          1.02
        ]
      ],
      "channelY": [
        [
          u,
          v,
          d
        ]
      ]
    },
    {
      "image_path": "relative-path-to-img2.png",
      "channelX": [
        [
          u,
          v,
          d
        ]
      ],
      "channelY": [
        [
          u,
          v,
          d,
        ]
      ]
    }
    ]
}
```

The number of channels (classes of keypoints) in the dataset is unlimited, but each datapoint must have the same channels and all channels must have a  number of keypoints defined as (u,v,[d]), being the (u,v) coord on the image plane and optionally the depth of the point w.r.t. the image plane.

For now, the (u,v) coordinates are expected to follow the blender convention:
- the origin is the left-down corner.
- U points left and is  in range [0,1]
- V points upwards and is in range [0,1]
However support for the native coordinate system (to which keypoints are now converted) is easily added by skipping the transformation (using an argmument parameter).

Note that a channel can have variable number of keypoints. This is useful if e.g. not all semantic keypoints of a class are always visible.

For an example, see the `test_dataset` at `test/test_dataset`.

## Training

There are 2 ways to train the keypoint detector:

The first is to run the `train.py` script with the appropriate arguments. e.g. from the root folder of this repo, you can run

```python keypoint_detection/train/train.py --keypoint_channels  "corner_keypoints flap_corner_keypoints" --keypoint_channel_max_keypoints "-1 -1" --image_dataset_path "/<path-to-workspace>/keypoint-detection/test/test_dataset" --json_dataset_path "<path-to-workspace>/keypoint-detection/test/test_dataset/dataset.json" --batch_size  1```

 to test on the provided test dataset, which contains 4 images. You should see the loss going down consistently until the detector has completely overfit the train set and the loss is around the entropy of the ground truth heatmaps (if you selected the default BCE loss).

The second method is to create a sweep on [wandb](https://wandb.ai) and to then start the sweep from the correct relative location.
A minimal sweep example  is given in `test/configuration.py`. The same content should be written to a yaml file according to the wandb format. The sweep can be started by running `wandb agent <sweep-id>` from your CLI.


### Train on your own dataset
To create your own configuration: run `python train.py -h` to see all parameter options and their documentation.

Provide at least the names of the keypoint classes (channels) in your dataset you want to train on as a string separated by a space, the max number of keypoints of each channel (again a space-separated string), the location of your dataset and your wandb configuration.


## Development

### vscode development container
For ease of use and in order to not pollute the OS, a [devcontainer]() configuration is included for vscode (see `.devcontainer`).

### submodules
- set to track main branch of keypoint detector
- update to latest commit on branch : `git submodule update --remote`
### Formatting, testing

- formatting and linting is done using [pre-commit](https://pre-commit.com/)
- testing is done using pytest (gh action for CI)


## Note on performance
- Not all parts of the codebase are extremely optimized. The functions to create heatmaps could be sped up probably.
- Keep in mind that the Average Precision is a very expensive operation, it can easily take as long to calculate the AP of a .1 split as it takes to train on the remaining 90% of the data. Therefore it makes sense to use them sparsely. The AP will always be calculated at the final epoch, so for optimal train performance (w/o intermediate feedback), you can e.g. set the `ap_epoch_start` parameter to your max number of epochs + 1.
