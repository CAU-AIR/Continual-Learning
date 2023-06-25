import argparse
import torch
import datetime as dt
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
import torch.optim.lr_scheduler
from avalanche.benchmarks import nc_benchmark
from avalanche.models.resnet32 import resnet32
from avalanche.training.supervised.strategy_wrappers import Replay
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.evaluation.metrics import ExperienceAccuracy, EpochAccuracy, StreamAccuracy
import random
import numpy as np


def main(args):
    device = 'cuda:' + args.device
    device = torch.device(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    dataset_stats = {
    'CIFAR10' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                 'size' : 32},
    'CIFAR100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
                 'size' : 32}
    }

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_stats['CIFAR100']['mean'], dataset_stats['CIFAR100']['std']),
        ]
    )
    test_transform = transforms.Compose(
        [
            ToTensor(), 
            transforms.Normalize(dataset_stats['CIFAR100']['mean'], dataset_stats['CIFAR100']['std'])
        ]
    )

    if args.dataset == 'CIFAR10':
        train_set = CIFAR10('data/CIFAR10', train=True, download=True, transform=train_transform)
        test_set = CIFAR10('data/CIFAR10', train=False, download=True, transform=test_transform)

    elif args.dataset == 'CIFAR100':
        train_set = CIFAR100('data/CIFAR100', train=True, download=True, transform=train_transform)
        test_set = CIFAR100('data/CIFAR100', train=False, download=True, transform=test_transform)


    benchmark = nc_benchmark(train_set, 
                             test_set, 
                             args.incremental, 
                             task_labels=False, 
                             seed=args.seed,
                             shuffle=False,
                             )

    # MODEL CREATION
    model = resnet32(num_classes=args.num_class)

    optim = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    sched = LRSchedulerPlugin(
        torch.optim.lr_scheduler.MultiStepLR(optim, [20,30,40,50], gamma=1.0 / 5.0)
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
        optim,
        criterion,
        mem_size=args.memory_size,
        train_epochs=args.epoch,
        train_mb_size=args.train_batch,
        eval_mb_size=args.eval_batch,
        device=device,
        plugins=[sched],
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--incremental', type=int, default=10)
    parser.add_argument('--memory_size', type=int, default=2000)
    parser.add_argument('--train_batch', type=int, default=2048)
    parser.add_argument('--eval_batch', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=60)

    args = parser.parse_args()

    main(args)