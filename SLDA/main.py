################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-04-2021                                                             #
# Author(s): Tyler Hayes                                                       #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import argparse
import torch
import warnings
import datetime as dt

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics,
)
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100
from avalanche.training.supervised.deep_slda import StreamingLDA
from avalanche.models import SLDAResNetModel
from avalanche.evaluation.metrics import ExperienceAccuracy, ExperienceLoss, ExperienceForgetting, ExperienceTime, EpochAccuracy

def main(args):
    # Device config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    print("device ", device)

    benchmark = SplitCIFAR10(5)

    # ---------
    date = dt.datetime.now()
    date = date.strftime("%Y_%m_%d_%H_%M_%S")
    tensor_logger = TensorboardLogger("SLDA/logs/CIFAR10/" + date)

    # eval_plugin = EvaluationPlugin(
    #     loss_metrics(epoch=True, experience=True, stream=True),
    #     accuracy_metrics(epoch=True, experience=True, stream=True),
    #     forgetting_metrics(experience=True, stream=True),
    #     loggers=[InteractiveLogger(), tensor_logger],
    # )

    interactive_logger = InteractiveLogger()
    tensor_logger = TensorboardLogger("logs_Jetson_iCaRL_cifar100_" + date)
    eval_plugin = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        ExperienceLoss(),
        ExperienceForgetting(),
        ExperienceTime(),
        loggers=[interactive_logger, tensor_logger])

    criterion = torch.nn.CrossEntropyLoss()
    model = SLDAResNetModel(
        device=device,
        arch="resnet18",
        imagenet_pretrained=args.imagenet_pretrained,
    )

    # CREATE THE STRATEGY INSTANCE
    cl_strategy = StreamingLDA(
        model,
        criterion,
        args.feature_size,
        args.n_classes,
        eval_mb_size=args.batch_size,
        train_mb_size=args.batch_size,
        train_epochs=1,
        shrinkage_param=args.shrinkage,
        streaming_update_sigma=args.plastic_cov,
        device=device,
        evaluator=eval_plugin,
    )

    warnings.warn(
        "The Deep SLDA example is not perfectly aligned with "
        "the paper implementation since it does not use a base "
        "initialization phase and instead starts streming from "
        "pre-trained weights."
    )

    # TRAINING LOOP
    print("Starting experiment...")
    for i, exp in enumerate(benchmark.train_stream):

        # fit SLDA model to batch (one sample at a time)
        cl_strategy.train(exp)

        # evaluate model on test data
        cl_strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLDA Example with ResNet-18 on CORe50")
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )

    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument(
        "--scenario",
        type=str,
        default="nc",
        choices=["ni", "nc", "nic", "nicv2_79", "nicv2_196", "nicv2_391"],
    )

    # deep slda model parameters
    parser.add_argument(
        "--imagenet_pretrained", type=bool, default=True
    )  # initialize backbone with
    # imagenet pre-trained weights
    parser.add_argument(
        "--feature_size", type=int, default=512 
    )  # feature size before output layer
    # (512 for resnet-18)
    parser.add_argument(
        "--shrinkage", type=float, default=1e-4
    )  # shrinkage value
    parser.add_argument(
        "--plastic_cov", type=bool, default=True
    )  # plastic covariance matrix
    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()
    main(args)