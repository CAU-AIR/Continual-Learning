import random
import argparse
import numpy as np
import datetime as dt

import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.benchmarks.datasets import CIFAR10, CIFAR100
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.models.resnet32 import resnet32
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (ExperienceAccuracy, StreamAccuracy, EpochAccuracy,)
from avalanche.logging import InteractiveLogger, TensorboardLogger


def main(args):
    device = 'cuda:' + args.device
    device = torch.device(device)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.enabled = False 
    cudnn.deterministic = True

    dataset_stats = {
    'CIFAR10' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                 'size' : 32},
    'CIFAR100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
                 'size' : 32}
    }

    transforms_group = dict(
        eval=(transforms.Compose(
                [
                    transforms.ToTensor(),
                ]),
            None,
        ),
        train=(transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
            None,
        ),
    )

    if args.dataset == 'CIFAR10':
        train_set = CIFAR10('data/CIFAR10', train=True, download=True)
        test_set = CIFAR10('data/CIFAR10', train=False, download=True)

    elif args.dataset == 'CIFAR100':
        train_set = CIFAR100('data/CIFAR100', train=True, download=True)
        test_set = CIFAR100('data/CIFAR100', train=False, download=True)

    train_set = make_classification_dataset(train_set, transform_groups=transforms_group, initial_transform_group="train",)
    test_set = make_classification_dataset(test_set, transform_groups=transforms_group, initial_transform_group="eval",)


    benchmark = nc_benchmark(train_set, 
                             test_set, 
                             args.incremental, 
                             task_labels=False, 
                             seed=args.seed,
                             shuffle=False,
                             fixed_class_order=args.fixed_class_order,
                             )

    # MODEL CREATION
    model = resnet32(num_classes=args.num_class)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    sched = LRSchedulerPlugin(
        torch.optim.lr_scheduler.MultiStepLR(optimizer, [20,30,40,50], gamma=1.0 / 5.0)
    )

    # choose some metrics and evaluation method
    date = dt.datetime.now()
    date = date.strftime("%Y_%m_%d_%H_%M_%S")

    interactive_logger = InteractiveLogger()
    tensor_logger = TensorboardLogger("ER/logs_er_" + args.dataset + "_" + date)

    # eval_plugin = EvaluationPlugin(
    #     accuracy_metrics(
    #         minibatch=True, epoch=True, experience=True, stream=True
    #     ),
    #     loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    #     forgetting_metrics(experience=True),
    #     loggers=[interactive_logger, tensor_logger],
    # )

    eval_plugin = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        StreamAccuracy(),
        loggers=[interactive_logger, tensor_logger])
    

    cl_strategy = Replay(
        model,
        optimizer,
        criterion,
        mem_size=args.memory_size,
        train_epochs=args.epoch,
        train_mb_size=args.train_batch,
        eval_mb_size=args.eval_batch,
        device=device,
        # plugins=[sched],
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")

    # # ocl_benchmark = OnlineCLScenario(batch_streams)
    # for i, exp in enumerate(benchmark.train_stream):
    #     cl_strategy.train(exp)
    #     cl_strategy.eval(benchmark.test_stream)

    for i, exp in enumerate(benchmark.train_stream):
        eval_exps = [e for e in benchmark.test_stream][: i + 1]
        cl_strategy.train(exp)
        cl_strategy.eval(eval_exps)


if __name__ == "__main__":
    fixed_class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--incremental', type=int, default=10)
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('--memory_size', type=int, default=2000)
    parser.add_argument('--train_batch', type=int, default=512)
    parser.add_argument('--eval_batch', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--fixed_class_order', type=list, default=fixed_class_order)

    args = parser.parse_args()

    main(args)