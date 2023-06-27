import random
import argparse
import numpy as np
import datetime as dt

import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.benchmarks.classic import ccub200, ccifar10, ccifar100
from avalanche.models.resnet32 import resnet32
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (ExperienceAccuracy, StreamAccuracy, EpochAccuracy,)
from avalanche.logging import InteractiveLogger, TensorboardLogger

from avalanche.training.supervised.strategy_wrappers import MIR
from avalanche.training.supervised import Naive
from avalanche.training.plugins import MIRPlugin, ReplayPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer


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
        args.num_class = 10
        fixed_class_order = np.arange(10)

        benchmark = ccifar10.SplitCIFAR10(
            n_experiences=args.incremental,
            seed=args.seed,
            fixed_class_order=fixed_class_order,
            dataset_root='data/CIFAR10'
        )

    elif args.dataset == 'CIFAR100':
        args.num_class = 100
        fixed_class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]

        benchmark = ccifar100.SplitCIFAR100(
            n_experiences=args.incremental,
            seed=args.seed,
            fixed_class_order=fixed_class_order,
            dataset_root='data/CIFAR100'
        )

    elif args.dataset == 'CUB200':
        args.num_class = 200

        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 244)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((224, 244)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        benchmark = ccub200.SplitCUB200(
            n_experiences=args.incremental,
            classes_first_batch=100,
            seed=args.seed,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root='data/CUB200'
        )

    model = resnet32(num_classes=args.num_class)
    if args.dataset == 'CUB200':
        model.fc = torch.nn.Linear(in_features=169344, out_features=args.num_class)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    sched = LRSchedulerPlugin(
        torch.optim.lr_scheduler.MultiStepLR(optimizer, [20,30,40,50], gamma=1.0 / 5.0)
    )

    # choose some metrics and evaluation method
    date = dt.datetime.now()
    date = date.strftime("%Y_%m_%d_%H_%M_%S")

    interactive_logger = InteractiveLogger()
    tensor_logger = TensorboardLogger("MIR/logs/" + args.dataset + "/" + args.device_name + "_" + date)
    eval_plugin = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        StreamAccuracy(),
        loggers=[interactive_logger, tensor_logger])

    storage_policy = ClassBalancedBuffer(args.memory_size, adaptive_size=True)
    replay_plugin = [ReplayPlugin(args.memory_size, storage_policy=storage_policy)]
    
    strategy = MIR(
        model = model,
        optimizer = optimizer,
        criterion = criterion,
        mem_size = args.memory_size,
        subsample=args.subsample_size,
        batch_size_mem=args.batch_size_mem,
        train_epochs=args.epoch,
        train_mb_size=args.train_batch,
        eval_mb_size=args.eval_batch,
        device=device,
        plugins=replay_plugin,
        evaluator=eval_plugin,
    )

    for i, exp in enumerate(benchmark.train_stream):
        eval_exps = [e for e in benchmark.test_stream][: i + 1]
        strategy.train(exp)
        strategy.eval(eval_exps)

    config = vars(args)
    metric_dict = eval_plugin.last_metric_results
    tensor_logger.writer.add_hparams(config, metric_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--device_name', type=str, default='cal_06')
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'CUB200'])
    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--incremental', type=int, default=10)
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.05)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--subsample_size', type=int, default=50)
    parser.add_argument('--batch_size_mem', type=int, default=10)
    parser.add_argument('--train_batch', type=int, default=10)
    parser.add_argument('--eval_batch', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=15)

    args = parser.parse_args()

    run_experiment(args)