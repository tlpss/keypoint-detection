"""quick and dirty benchmark of the heatmap extraction methods."""

import time

import torch

from keypoint_detection.utils.heatmap import (
    generate_channel_heatmap,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_scipy,
)


def test_method(nb_iters, heatmaps, method, name):
    n_keypoints = 20
    torch.cuda.synchronize()
    t0 = time.time()
    if method == get_keypoints_from_heatmap_scipy:
        for i in range(nb_iters):
            heatmap = heatmaps[i]
            for batch in range(len(heatmap)):
                for channel in range(len(heatmap[batch])):
                    method(heatmap[batch][channel], n_keypoints)
    else:
        for i in range(nb_iters):
            method(heatmaps[i], n_keypoints)
    torch.cuda.synchronize()
    t1 = time.time()
    duration = (t1 - t0) / nb_iters * 1000.0
    print(f"{duration:.3f} ms per iter for {name} method with heatmap size {heatmap_size} ")


if __name__ == "__main__":
    nb_iters = 20
    n_channels = 2
    batch_size = 1
    n_keypoints_per_channel = 10
    print(
        f"benchmarking with  batch_size: {batch_size}, {n_channels} channels and {n_keypoints_per_channel} keypoints per channel"
    )
    for heatmap_size in [(256, 256), (512, 256), (512, 512), (1920, 1080)]:
        heatmaps = [
            generate_channel_heatmap(heatmap_size, torch.randint(0, 255, (6, 2)), 6, "cpu")
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, n_channels, 1, 1)
            .cuda()
            for _ in range(nb_iters)
        ]

        test_method(nb_iters, heatmaps, get_keypoints_from_heatmap_scipy, "scipy")
        test_method(nb_iters, heatmaps, get_keypoints_from_heatmap_batch_maxpool, "torch")
