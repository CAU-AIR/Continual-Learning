import argparse
import torch
import random
import numpy as np
import datetime as dt
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from utils.util import icarl_cifar100_augment_data, get_dataset_per_pixel_mean

from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.supervised import ICaRL
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.benchmarks.datasets import CIFAR10, CIFAR100
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset, make_classification_dataset
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.evaluation.metrics import ExperienceAccuracy, EpochAccuracy, StreamAccuracy


def run_experiment(args):
    device = 'cuda:' + args.device
    device = torch.device(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    if args.dataset == 'CIFAR10':
        train_set = CIFAR10('data/CIFAR10', train=True, download=True)
        test_set = CIFAR10('data/CIFAR10', train=False, download=True)

        per_pixel_mean = get_dataset_per_pixel_mean(CIFAR10('data/CIFAR10', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])))

    elif args.dataset == 'CIFAR100':
        train_set = CIFAR100('data/CIFAR100', train=True, download=True)
        test_set = CIFAR100('data/CIFAR100', train=False, download=True)

        per_pixel_mean = get_dataset_per_pixel_mean(CIFAR100('data/CIFAR100', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])))

    transforms_group = dict(
        eval=(transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img_pattern: img_pattern - per_pixel_mean,
                ]),
            None,
        ),
        train=(transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img_pattern: img_pattern - per_pixel_mean,
                    icarl_cifar100_augment_data,
                ]),
            None,
        ),
    )

    train_set = make_classification_dataset(train_set, transform_groups=transforms_group, initial_transform_group="train",)
    test_set = make_classification_dataset(test_set, transform_groups=transforms_group, initial_transform_group="eval",)

    lr_milestones = [20,30,40,50]
    lr_factor = 5.0

    if args.dataset == 'CIFAR10':
        scenario = nc_benchmark(train_dataset=train_set,
                            test_dataset=test_set,
                            n_experiences=args.incremental,
                            task_labels=False,
                            seed=args.seed,
                            shuffle=False,
                            )
    elif args.dataset == 'CIFAR100':
        fixed_class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]

        scenario = nc_benchmark(train_dataset=train_set,
                            test_dataset=test_set,
                            n_experiences=args.incremental,
                            task_labels=False,
                            seed=args.seed,
                            shuffle=False,
                            fixed_class_order=fixed_class_order
                            )

    model: IcarlNet = make_icarl_net(num_classes=args.num_class)
    model.apply(initialize_icarl_net)

    optimizer = optim.SGD(model.parameters(), lr=2.0, momentum=0.9, weight_decay=1e-5)
    sched = LRSchedulerPlugin(MultiStepLR(optimizer, lr_milestones, gamma=1.0 / lr_factor))

    date = dt.datetime.now()
    date = date.strftime("%Y_%m_%d_%H_%M_%S")

    interactive_logger = InteractiveLogger()
    tensor_logger = TensorboardLogger("iCaRL/logs_iCaRL_" + args.dataset + "_" + date)
    eval_plugin = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        StreamAccuracy(),
        loggers=[interactive_logger, tensor_logger])
    
    buffer_transform = transforms.Compose([icarl_cifar100_augment_data])

    strategies = ICaRL(model.feature_extractor, model.classifier, optimizer, args.memory_size, buffer_transform=buffer_transform, fixed_memory=True, train_mb_size=args.train_batch, train_epochs=args.epoch, eval_mb_size=args.eval_batch, device=device, plugins=[sched], evaluator=eval_plugin)  # criterion = ICaRLLossPlugin()

    for i, exp in enumerate(scenario.train_stream):
        eval_exps = [e for e in scenario.test_stream][: i + 1]
        strategies.train(exp)
        strategies.eval(eval_exps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--incremental', type=int, default=10)
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('--memory_size', type=int, default=2000)
    parser.add_argument('--train_batch', type=int, default=2048)
    parser.add_argument('--eval_batch', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=60)

    args = parser.parse_args()

    run_experiment(args)