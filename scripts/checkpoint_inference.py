"""example script for inference on local image with a saved model checkpoint"""

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint


def local_inference(model, image: np.ndarray, device="cuda"):
    """inference on a single image as if you would load the image from disk or get it from a camera.
    Returns a list of the extracted keypoints for each channel.


    """
    # assert model is in eval mode! (important for batch norm layers)
    assert model.training == False, "model should be in eval mode for inference"

    # convert image to tensor with correct shape (channels, height, width) and convert to floats in range [0,1]
    # add batch dimension
    # and move to device
    image = to_tensor(image).unsqueeze(0).to(device)

    # pass through model
    with torch.no_grad():
        heatmaps = model(image).squeeze(0)

    # extract keypoints from heatmaps
    predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps.unsqueeze(0))[0]

    return predicted_keypoints


if __name__ == "__main__":
    import pathlib

    from skimage import io

    """example for loading models to run inference from a pytorch lightning checkpoint

    for faster inference you probably want to consider
    - reducing the model precision (mixed precision or simply half precision) on new GPUs with TensorCores
            but do not forget to set the cudnn benchmark https://github.com/pytorch/pytorch/issues/46377
    - compiling the model to torchscript
    - or compiling it with Pytorch 2.0 (not yet released)
    - or using TensorRT

    see benchmark.py script for how to test the influences inference speed, which is ofc determined by the model (size), the input size and your hardware.
    """

    checkpoint_name = "airo-box-manipulation/keypoint-detector-integration-test/model-q360zo4y:v29"
    image_path = pathlib.Path("").parent / "test" / "test_dataset" / "images" / "0.png"

    # load a wandb checkpoint
    model = get_model_from_wandb_checkpoint(checkpoint_name)

    # do not forget to set model to eval mode!
    # this will e.g. use the running statistics for batch norm layers instead of the batch statistics.
    # this is important as inference batches are typically a lot smaller which would create too much noise.
    model.eval()

    # move model to gpu
    model.cuda()

    # load image from disk
    # although at inference you will most likely get the image from the camera
    # format will be the same though:
    #  (height, width, channels) ints in range [0,255]
    # beware of the color channels order, it should be RGBD.
    image = io.imread(image_path)

    keypoints = local_inference(model, image)
    print(keypoints)
