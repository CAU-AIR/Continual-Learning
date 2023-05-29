from os.path import expanduser

import torch

from avalanche.benchmarks.datasets import CIFAR100
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import SlimResNet18
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim import SGD
from torchvision import transforms
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (ExperienceAccuracy,StreamAccuracy,EpochAccuracy,)
from avalanche.logging.interactive_logging import InteractiveLogger
import random
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from train.supervised.strategy_wrappers import MIR


def run_experiment(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    transforms_group = dict(
        eval=(
            transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            None,
        ),
        train=(
            transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            None,
        ),
    )

    train_set = CIFAR100(
        "data/cifar100/",
        train=True,
        download=True,
    )
    test_set = CIFAR100(
        "data/cifar100/",
        train=False,
        download=True,
    )

    train_set = AvalancheDataset(
        train_set,
        transform_groups=transforms_group,
        initial_transform_group="train",
    )
    test_set = AvalancheDataset(
        test_set,
        transform_groups=transforms_group,
        initial_transform_group="eval",
    )

    scenario = nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=config.nb_exp,
        task_labels=False,
        seed=config.seed,
        shuffle=False,
        fixed_class_order=config.fixed_class_order,
    )

    evaluator = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        StreamAccuracy(),
        loggers=[InteractiveLogger()],
    )

    model = SlimResNet18(nclasses=100)

    optim = SGD(
        model.parameters(),
        lr=config.lr_base,
        weight_decay=config.wght_decay,
        momentum=0.9,
    )
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, config.lr_milestones, gamma=1.0 / config.lr_factor)
    )
    criterion = torch.nn.CrossEntropyLoss()

    strategy = MIR(
        model = model,
        optimizer = optim,
        criterion = criterion,
        mem_size = config.memory_size,
        subsample=50,
        train_mb_size=config.batch_size,
        train_epochs=config.epochs,
        eval_mb_size=config.batch_size,
        device=device,
        plugins=[sched],
        evaluator=evaluator,
   )

    for i, exp in enumerate(scenario.train_stream):
        eval_exps = [e for e in scenario.test_stream][: i + 1]
        strategy.train(exp, num_workers=4)
        strategy.eval(eval_exps, num_workers=4)


class Config(dict):
    def __getattribute__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == "__main__":
    config = Config()

    config.batch_size = 1
    config.nb_exp = 10
    config.memory_size = 2000
    config.epochs = 1
    config.lr_base = 0.01
    config.lr_milestones = [49, 63]
    config.lr_factor = 5.0
    config.wght_decay = 0.00001
    config.fixed_class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]
    config.seed = 2222

    run_experiment(config)