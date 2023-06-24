import argparse
import torch
import datetime as dt
from torchvision import transforms
from torchvision.datasets import CIFAR100
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


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    n_batches = 10  # split classes
    # ---------

    # --- TRANSFORMATIONS
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
    # ---------

    # --- SCENARIO CREATION
    cifar_train = CIFAR100(
        root="data/cifar100/",
        train=True,
        download=True,
        transform=train_transform,
    )
    cifar_test = CIFAR100(
        root="data/cifar100/",
        train=False,
        download=True,
        transform=test_transform,
    )
    benchmark = nc_benchmark(
        cifar_train, cifar_test, n_batches, task_labels=False, seed=0
    )
    # ---------

    # MODEL CREATION
    model = resnet32(num_classes=benchmark.n_classes)

    # choose some metrics and evaluation method
    date = dt.datetime.now()
    date = date.strftime("%Y_%m_%d_%H_%M_%S")

    interactive_logger = InteractiveLogger()
    tensor_logger = TensorboardLogger("ER/logs_er_cifar100_" + date)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger, tensor_logger],
    )

    optim = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    sched = LRSchedulerPlugin(
        torch.optim.lr_scheduler.MultiStepLR(optim, [20,30,40,50], gamma=1.0 / 5.0)
    )

    cl_strategy = Replay(
        model,
        optim,
        criterion,
        mem_size=2000,
        train_mb_size=512,
        train_epochs=60,
        eval_mb_size=256,
        device=device,
        plugins=[sched],
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []

    # ocl_benchmark = OnlineCLScenario(batch_streams)
    for i, exp in enumerate(benchmark.train_stream):
        cl_strategy.train(exp)
        results.append(cl_strategy.eval(benchmark.test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)