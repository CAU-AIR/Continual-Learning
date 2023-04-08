import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import dataloader
import data_generator
from models.IcarlNet import icarl_net
from metric import AverageMeter, Logger

parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--device_name', type=str, default='hspark')
# Dataset Settings
parser.add_argument('--root', type=str, default='./data/')
parser.add_argument('--dataset', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'HAR'])
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=0)
# Model Settings
parser.add_argument('--model_name', type=str, default='iCaRL', choices=['iCaRL', 'ImageNet_ResNet18'])
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--classifier', type=str, default='FC', choices=['FC', 'NCM'])
# CL Settings
parser.add_argument('--class_increment', type=int, default=1)

args = parser.parse_args()

def main():
    ## GPU Setup
    device = 'cuda:' + args.device
    args.device = torch.device(device)
    torch.cuda.set_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Dataset Generator
    if 'CIFAR' in args.dataset:
        data_generator.__dict__['Generator'](args)
        if args.dataset == 'CIFAR10': args.num_classes = 10
        else: args.num_classes = 100

    # Create Model
    model_name = args.model_name