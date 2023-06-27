import os
import sys
import math
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import dataloader
import data_generator
from models import IcarlNet, NCM
from icarl import ICaRLPlugin
from utils.losses import ICaRLLossPlugin
from utils.util import Logger, AverageMeter, accuracy

parser = argparse.ArgumentParser()
# General Settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--device_name', type=str, default='hspark')
parser.add_argument('--log_path', type=str, default='')
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
parser.add_argument('--lamb', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--classifier', type=str, default='NCM', choices=['FC', 'NCM'])
# CL Settings
parser.add_argument('--class_increment', type=int, default=1)
parser.add_argument('--fixed_order', action='store_true')
parser.add_argument('--memory', type=int, default=100)

args = parser.parse_args()

def train(task, epoch, model, strategy, train_loader, criterion, optimizer, classifier=None):
    acc = AverageMeter()
    losses = AverageMeter()
    num_iter = math.ceil(len(train_loader.dataset) / args.batch_size)

    if task > 0:
        memory_transform = dataloader.get_transform(args.dataset)
        memory_dataset = dataloader.memory_dataset(strategy.x_memory, strategy.y_memory, memory_transform)
        memory_loader = DataLoader(memory_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        memory_iter = iter(memory_loader)

    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        # y = y.type(torch.LongTensor)
        x, y = x.to(args.device).float(), y.to(args.device)

        if task > 0:
            try:
                print('\n', x.size())
                x_memory, y_memory = next(memory_iter)
                print(x_memory.size())
                x = torch.cat([x, x_memory])
                print(x.size())
                y = torch.cat([y, y_memory.to(args.device)])
            except StopIteration:
                pass
            criterion.before_forward(x) # set old model logits

        if args.classifier == 'NCM':
            features = model.features(x)
            classifier.train_(features, y)
            logits = classifier.evaluate_(features)
        else:
            logits = model(x) # FC

        loss = criterion(logits, y)
        acc1 = accuracy(logits, y)

        # Compute Gradient and do SGD step
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        losses.update(loss)
        acc.update(acc1[0], x.size(0))

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.2f Accuracy: %.2f' % (args.dataset, epoch+1, args.epoch, batch_idx+1, num_iter, loss, acc.avg))
        sys.stdout.flush()
    
        criterion.after_training_exp(model, y)

    return loss.item(), acc.avg

def test(task, model, test_loader, classifier=None):
    acc = AverageMeter()
    sys.stdout.write('\n')

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device).float(), y.to(args.device)

            if args.classifier == 'NCM':
                features = model.features(x)
                logits = classifier(features)
            else:
                features = model.features(x)
                logits = classifier(features)

            acc1 = accuracy(logits, y, test=True)
            acc.update(acc1[0], x.size(0))

            sys.stdout.write('\r')
            sys.stdout.write("Test | Accuracy (Test Dataset Up to Task-%d): %.2f%%" % (task+1, acc.avg))
            sys.stdout.flush()

    return acc.avg

def main():
    ## GPU Setup
    if args.device == "mps":
        device = args.device # M1 (MacBook)
        args.device = torch.device(device)
    else:
        device = 'cuda:' + args.device # GPU
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
    if 'iCaRL' in model_name:
        model = IcarlNet.__dict__[args.model_name](args.num_classes)
        model.apply(IcarlNet.initialize_icarl_net)
        model.to(args.device)

    # Optimizer and Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[49, 63], gamma=1.0 / 5.0)
    criterion = ICaRLLossPlugin(device)

    feature_size = model.input_dims
    classifier_name = args.classifier
    
    if classifier_name == 'NCM':
        print('classifier : ', classifier_name)
        train_classifier = NCM.NearestClassMean(feature_size, args.num_classes, device=args.device)
    else:
        print('classifier : ', classifier_name)
        train_classifier=None

    icarl = ICaRLPlugin(args, model, feature_size, device)

    # For plotting the logs
    args.log_path = os.path.dirname(os.path.realpath(__file__))
    logger = Logger(args.log_path + '/logs/' + args.dataset + '/' + args.device_name, args.classifier)
    log_t = 1

    data_loader = dataloader.dataloader(args)
    last_test_acc = 0

    task_num = 0
    fixed_class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]


    for idx in range(0, args.num_classes, args.class_increment):
        if args.fixed_order:
            task = fixed_class_order[idx:idx+args.class_increment]
        else:
            task = [k for k in range(idx, idx+args.class_increment)]
        print('\nTask Index : ', task_num)
        print('\nTask : ', task)
        icarl.observed_classes = task_num + 1

        train_loader = data_loader.load(task)
        test_loader = data_loader.load(task, train=False)

        best_acc = 0
        for epoch in range(args.epoch):
            loss, train_acc = train(task_num, epoch, model, icarl, train_loader, criterion, optimizer, train_classifier)

            if train_acc > best_acc:
                best_acc = train_acc
                logger.result('Train Epoch Loss/Labeled', loss, epoch)

            # if classifier_name == 'NCM' and epoch+1 != args.epoch:
            icarl.after_training_exp(train_loader)

            scheduler.step()

        logger.result('Train Accuracy', best_acc, log_t)

        test_acc = test(task_num, model, test_loader, icarl)
        logger.result('Test Accuracy', test_acc, log_t)
        last_test_acc = test_acc

        task_num += 1
        log_t += 1

    logger.result('Final Test Accuracy', last_test_acc, 1)
    print("\n\nFinal Test Accuracy : %.2f%%" % last_test_acc)

    args.device = device
    metric_dict = {'metric': last_test_acc}
    logger.config(config=args, metric_dict=metric_dict)

if __name__ == '__main__':
    main()