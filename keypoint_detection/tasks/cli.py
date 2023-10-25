"""cli entry point"""
import sys

from keypoint_detection.tasks.eval import eval_cli
from keypoint_detection.tasks.train import train_cli

TRAIN_TASK = "train"
EVAL_TASK = "eval"
TASKS = [TRAIN_TASK, EVAL_TASK]


def main():
    # read command line args in plain python

    # TODO this is a very hacky approach for combining independent cli scripts
    # should redesign this in the future.

    print(sys.argv)
    task = sys.argv[1]
    sys.argv.pop(1)

    if task == "--help" or task == "-h":
        print("Usage: keypoint-detection [task] [task args]")
        print(f"Tasks: {TASKS}")
    elif task == TRAIN_TASK:
        train_cli()
    elif task == EVAL_TASK:
        eval_cli()
