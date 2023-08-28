import torch


def benchmark(f, name=None, iters=500, warmup=20, display=True, profile=False):
    """Pytorch Benchmark script, copied from Horace He at https://gist.github.com/Chillee/f86675147366a7a0c6e244eaa78660f7#file-2-overhead-py"""
    import time

    # warmup as some operations are initialized lazily
    # or some optimizations still need to happen
    for _ in range(warmup):
        f()
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    torch.cuda.synchronize()  # wait for all kernels to finish
    begin = time.time()
    for _ in range(iters):
        f()
    torch.cuda.synchronize()  # wait for all kernels to finish
    us_per_iter = (time.time() - begin) * 1e6 / iters
    if name is None:
        res = us_per_iter
    else:
        res = f"{name}: {us_per_iter:.2f}us / iter"
    if display:
        print(res)
    return res


if __name__ == "__main__":
    """example code for benchmarking model/inference speed"""
    import numpy as np
    from checkpoint_inference import local_inference

    from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
    from keypoint_detection.models.detector import KeypointDetector

    device = "cuda:0"
    backbone = "ConvNeXtUnet"
    input_size = 512

    backbone = BackboneFactory.create_backbone(backbone)
    model = KeypointDetector(1, "2 4", 3, 3e-4, backbone, [["test1"], ["test2,test3"]], 1, 1, 0.0, 20)
    # do not forget to set model to eval mode!
    # this will e.g. use the running statistics for batch norm layers instead of the batch statistics.
    # this is important as inference batches are typically a lot smaller which would create too much noise.
    model.eval()
    model.to(device)

    sample_model_input = torch.rand(1, 3, input_size, input_size, device=device, dtype=torch.float32)
    sample_inference_input = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

    benchmark(lambda: model(sample_model_input), "plain model forward pass", profile=False)
    benchmark(
        lambda: local_inference(model, sample_inference_input, device=device), "plain model inference", profile=False
    )

    torchscript_model = model.to_torchscript()
    # JIT compiling with torchscript should improve performance (slightly)
    benchmark(lambda: torchscript_model(sample_model_input), "torchscript model forward pass", profile=False)

    torch.backends.cudnn.benchmark = True
    model.half()
    half_input = sample_model_input.half()
    half_torchscript_model = model.to_torchscript(method="trace", example_inputs=half_input)

    benchmark(
        lambda: half_torchscript_model(half_input), "torchscript model forward pass with half precision", profile=False
    )

    # note: from the traces it can be seen that a lot of time is spent in 'overhead', i.e. the GPU is idle...
    # so compiling the model should provide huge boosts in inference speed.
