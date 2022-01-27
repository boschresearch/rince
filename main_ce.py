from __future__ import print_function

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from datasets.customCifar import CIFAR10, CIFAR100
from networks.resnet_big import SupCEResNet
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import str2bool


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--size', type=int, default=32, help='parameter for crop')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=None)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    # parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--exp_name', type=str, default=None, help='set experiment name manually')
    parser.add_argument('--use_ssl_augmentations', type=str2bool, default='False')
    opt = parser.parse_args()

    if opt.seed:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.exp_name:
        opt.model_name = opt.exp_name

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'imagenet':
        opt.n_cls = 1000
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def check_exists(path1, path2):
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        raise Exception('Fodler not found')


def set_loader(opt, retrieval=False, labelset='fine', overwrite_mean_and_std_dataset=None):
    # construct data loader
    if not overwrite_mean_and_std_dataset is None:
        tmp = opt.dataset
        opt.dataset = overwrite_mean_and_std_dataset
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'imagenet' or opt.dataset.lower() == 'tiny_imagenet' or opt.dataset == 'tiny_imagenet_inliers' or opt.dataset == 'imagenet_100':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    if not overwrite_mean_and_std_dataset is None:
        opt.dataset = tmp
    val_datasets = None
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.use_ssl_augmentations:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    if not retrieval:
        if (opt.dataset == 'imagenet_100' or opt.dataset == 'AwA2') and not opt.dataset == 'tiny_imagenet':
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(opt.size),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    else:
        if (opt.dataset == 'imagenet_100' or opt.dataset == 'AwA2' or opt.dataset == 'imagenet') and not opt.dataset == 'tiny_imagenet':
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(opt.size),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            val_transform = transforms.Compose([
                transforms.Resize(size=opt.size),
                transforms.ToTensor(),
                normalize,
            ])
    if retrieval:
        train_transform = val_transform

    if opt.dataset == 'cifar10':
        if not retrieval:
            train_dataset = CIFAR10(root=opt.data_folder,
                                    transform=train_transform,
                                    download=True,
                                    labelset=labelset)
        else:
            train_dataset = CIFAR10(root=opt.data_folder,
                                    transform=val_transform,
                                    download=True,
                                    labelset=labelset)
        val_dataset = CIFAR10(root=opt.data_folder,
                              train=False,
                              transform=val_transform,
                              labelset=labelset)
    elif opt.dataset == 'cifar100':
        if not retrieval:
            train_dataset = CIFAR100(root=opt.data_folder,
                                     transform=train_transform,
                                     download=True,
                                     labelset=labelset)
        else:
            train_dataset = CIFAR100(root=opt.data_folder,
                                     transform=val_transform,
                                     download=True,
                                     labelset=labelset)
        val_dataset = CIFAR100(root=opt.data_folder,
                               train=False,
                               transform=val_transform,
                               labelset=labelset)

    elif opt.dataset == 'imagenet' or opt.dataset.lower() == 'tiny_imagenet' or opt.dataset == 'tiny_imagenet_inliers' or opt.dataset == 'imagenet_100':
        if not retrieval:
            train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                                 transform=train_transform)
        else:
            train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                                 transform=val_transform)

        val_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'val'),
                                           transform=val_transform)
    elif opt.dataset == 'AwA2':
        if not retrieval:
            train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder),
                                                 transform=train_transform)
        else:
            train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder),
                                                 transform=val_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder),
                                           transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    if not opt.dataset == 'AwA2':
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    else:
        train_loader = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    label_dict = {}
    for i in range(20):
        label_dict[i] = []
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)

        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            if opt.two_heads:
                output = output[0]
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt, labelset='fine')

    # build model and criterion
    model, criterion = set_model(opt)
    val_criterion = criterion

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    tb_writer = SummaryWriter(log_dir=opt.tb_path)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        tb_writer.add_scalar('train_loss', loss, epoch)
        tb_writer.add_scalar('train_acc', train_acc, epoch)
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc = validate(val_loader, model, val_criterion, opt)
        tb_writer.add_scalar('val_loss', loss, epoch)
        tb_writer.add_scalar('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
