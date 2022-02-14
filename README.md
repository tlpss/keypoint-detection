# keypoint-detection
2D keypoint detection with CNN using Pytorch Lightning and wandb


## Dataset format

This package expects a dataset with the following format:

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

The number of channels in the dataset is unlimited, but each datapoint must have the same channels and all channels must have a  number of keypoints defined as (u,v,d) being the (u,v) coord on the image plane and optionally the depth of the point w.r.t. the image plane.

Note that a channel can have variable number of keypoints.
