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
from avalanche.training.supervised.strategy_wrappers import MIR

def run_experiment(args):
    device = 'cuda:' + args.device
    device = torch.device(device)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.enabled = False 
    cudnn.deterministic = True

    if args.dataset == 'CIFAR10':
        train_set = CIFAR10('data/CIFAR10', train=True, download=True)
        test_set = CIFAR10('data/CIFAR10', train=False, download=True)

    elif args.dataset == 'CIFAR100':
        train_set = CIFAR100('data/CIFAR100', train=True, download=True)
        test_set = CIFAR100('data/CIFAR100', train=False, download=True)

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

    train_set = make_classification_dataset(train_set, transform_groups=transforms_group, initial_transform_group="train",)
    test_set = make_classification_dataset(test_set, transform_groups=transforms_group, initial_transform_group="eval",)

    scenario = nc_benchmark(train_dataset=train_set,
                        test_dataset=test_set,
                        n_experiences=args.incremental,
                        task_labels=False,
                        seed=args.seed,
                        shuffle=False,
                        fixed_class_order=args.fixed_class_order
                        )

    model = resnet32(num_classes=args.num_class)

    lr_milestones = [20,30,40,50]
    lr_factor = 5.0
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    # sched = LRSchedulerPlugin(MultiStepLR(optimizer, lr_milestones, gamma=1.0 / lr_factor))
    criterion = torch.nn.CrossEntropyLoss()

    date = dt.datetime.now()
    date = date.strftime("%Y_%m_%d_%H_%M_%S")

    interactive_logger = InteractiveLogger()
    tensor_logger = TensorboardLogger("MIR/logs_mir_" + args.dataset + "_" + date)
    eval_plugin = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        StreamAccuracy(),
        loggers=[interactive_logger, tensor_logger])

    strategy = MIR(
        model = model,
        optimizer = optimizer,
        criterion = criterion,
        mem_size = args.memory_size,
        subsample=args.subsample_size,
        train_epochs=args.epoch,
        train_mb_size=args.train_batch,
        eval_mb_size=args.eval_batch,
        device=device,
        # plugins=[sched],
        evaluator=eval_plugin,
   )

    for i, exp in enumerate(scenario.train_stream):
        eval_exps = [e for e in scenario.test_stream][: i + 1]
        strategy.train(exp)
        strategy.eval(eval_exps)


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
    parser.add_argument('--subsample_size', type=int, default=50)
    parser.add_argument('--train_batch', type=int, default=512)
    parser.add_argument('--eval_batch', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--fixed_class_order', type=list, default=fixed_class_order)

    args = parser.parse_args()

    run_experiment(args)