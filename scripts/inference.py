import torch

from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint


def benchmark(model):
    """
    unreliable benchmark but jit and no_Grad are important!
    """

    # warmup?
    print(timeit.timeit(lambda: model(torch.rand(1, 3, 256, 256, device="cuda")), number=1000))

    # start benchmark
    with torch.no_grad():
        print(timeit.timeit(lambda: model(torch.rand(1, 3, 256, 256, device="cuda")), number=1000))

    print(timeit.timeit(lambda: model(torch.rand(1, 3, 256, 256, device="cuda")), number=1000))

    model.to_torchscript()
    print(timeit.timeit(lambda: model(torch.rand(1, 3, 256, 256, device="cuda")), number=1000))
    with torch.no_grad():
        print(timeit.timeit(lambda: model(torch.rand(1, 3, 256, 256, device="cuda")), number=1000))


if __name__ == "__main__":
    import timeit

    """example for loading models to run inference from a pytorch lightning checkpoint
    """

    checkpoint = "airo-box-manipulation/keypoint-detector-integration-test/model-22188iqq:v4"

    model = get_model_from_wandb_checkpoint(checkpoint)
    # do not forget to set model to eval mode!
    # this will e.g. use the running statistics for batch norm layers instead of the batch statistics.
    # this is important as inference batches are typically a lot smaller which would create too much noise.
    model.eval()
    model.cuda()
    benchmark(model)

    model.to_torchscript()
    # for faster inference: consider using TensorRT
