import argparse
import torch
import random
import numpy as np
import datetime as dt
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.benchmarks.classic import ccub200, ccifar10, ccifar100
from avalanche.benchmarks import datasets
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.evaluation.metrics import ExperienceAccuracy, EpochAccuracy, StreamAccuracy

from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.supervised import ICaRL
from utils.util import icarl_cifar_augment_data, get_dataset_per_pixel_mean


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

    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        data_root = 'data/' + args.dataset
        per_pixel_mean = get_dataset_per_pixel_mean(datasets.__dict__[args.dataset](data_root, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])))

        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                lambda img_pattern: img_pattern - per_pixel_mean,
                icarl_cifar_augment_data,
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                lambda img_pattern: img_pattern - per_pixel_mean,
            ]
        )

    if args.dataset == 'CIFAR10':
        args.num_class = 10
        fixed_class_order = [i for i in range(10)]

        benchmark = ccifar10.SplitCIFAR10(
            n_experiences=args.incremental,
            seed=args.seed,
            train_transform=train_transform,
            eval_transform=eval_transform,
            fixed_class_order=fixed_class_order,
            dataset_root='data/CIFAR10'
        )

    elif args.dataset == 'CIFAR100':
        args.num_class = 100
        fixed_class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]

        benchmark = ccifar100.SplitCIFAR100(
            n_experiences=args.incremental,
            seed=args.seed,
            train_transform=train_transform,
            eval_transform=eval_transform,
            fixed_class_order=fixed_class_order,
            dataset_root='data/CIFAR100'
        )

    elif args.dataset == 'CUB200':
        args.num_class = 200

        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
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
    

    model: IcarlNet = make_icarl_net(num_classes=args.num_class)
    model.apply(initialize_icarl_net)

    lr_milestones = [20,30,40,50]
    lr_factor = 5.0
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    sched = LRSchedulerPlugin(MultiStepLR(optimizer, lr_milestones, gamma=1.0 / lr_factor))

    date = dt.datetime.now()
    date = date.strftime("%Y_%m_%d_%H_%M_%S")

    interactive_logger = InteractiveLogger()
    tensor_logger = TensorboardLogger("iCaRL/logs/" + args.dataset + "/" + args.device_name + "_" + date)
    eval_plugin = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        StreamAccuracy(),
        loggers=[interactive_logger, tensor_logger])
    
    buffer_transform = transforms.Compose([icarl_cifar_augment_data])

    strategy = ICaRL(
        model.feature_extractor, 
        model.classifier, 
        optimizer, 
        args.memory_size, 
        buffer_transform=buffer_transform, 
        fixed_memory=True, 
        train_mb_size=args.train_batch, 
        train_epochs=args.epoch, 
        eval_mb_size=args.eval_batch, 
        device=device, 
        plugins=[sched], 
        evaluator=eval_plugin,
    )  # criterion = ICaRLLossPlugin()


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
    parser.add_argument('--device_name', type=str, default='cal_05')
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'CUB200'])
    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--incremental', type=int, default=10)
    parser.add_argument('--lr', '--learning_rate', type=float, default=2.)
    parser.add_argument('--memory_size', type=int, default=2000)
    parser.add_argument('--train_batch', type=int, default=512)
    parser.add_argument('--eval_batch', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=60)
 
    args = parser.parse_args()

    run_experiment(args)