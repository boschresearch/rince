from __future__ import print_function

import argparse
import math
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from losses import ContrastiveRanking
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from util import str2bool

tr = torchvision.models.wide_resnet50_2()

import sys

sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), 'AutoAugment'))
try:
    from autoaugment import ImageNetPolicy
except ImportError:
    print('AutoAugment not found, only standard augmentation available')


# note that results in paper have been obtained with a single GPU. Support for more GPUs only available for convenience without any gurantees. Code was not tested with GPUs distributed across nodes.
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--ranking_baseline', default=False, type=str2bool, help="run ranking baseline")
    parser.add_argument('--minimize_class_distance', default=True, type=str2bool, help='minimize same class distance')
    parser.add_argument('--remove_ranking_between_rank1_rank2', default=False, type=str2bool,
                        help='when --minimize_class_distance is set we can optionally remove the first ranking loss')
    parser.add_argument('--topk_negatives', default=-1, type=int, help='you topk nearest negatives as hard negatives')
    # stuff for ranking
    parser.add_argument('--min_tau', default=0.1, type=float, help='min temperature parameter in SimCLR')
    parser.add_argument('--max_tau', default=0.2, type=float, help='max temperature parameter in SimCLR')
    parser.add_argument('--m', default=0.99, type=float, help='momentum update to use in contrastive learning')
    parser.add_argument('--supervised_mode', type=str, default='class_simmilarity_ranking')
    parser.add_argument('--do_sum_in_log', type=str2bool, default='True')
    parser.add_argument('--memorybank_size', default=4096, type=int)

    parser.add_argument('--similarity_threshold', default=0.01, type=float, help='')
    parser.add_argument('--sample_n_simclasses', default='False', type=str2bool)
    parser.add_argument('--use_dynamic_tau', type=str2bool, default='True', help='')
    parser.add_argument('--use_supercategories', type=str2bool, default='False', help='')
    parser.add_argument('--similarity_ranking_imagenet', type=str, default='sibling',
                        help='which type of similarity ranking to use; available: (sibling, class, hierarchy_k)')
    parser.add_argument('--hierarchy_k', type=int, default=2)
    parser.add_argument('--mixed_out_in', type=str2bool, default='False')
    parser.add_argument("--nameSimPath", type=str, default='./word_sims_imagenet100.npy')
    parser.add_argument("--path_name2wordnet", type=str, default='./ImageNet100_name2wordnetId.npy')

    parser.add_argument('--exp_name', type=str, default=None, help='set experiment name manually')
    parser.add_argument('--mixed_out_in_log', type=str2bool, default='False', help='')
    parser.add_argument('--out_in_log', type=str2bool, default='False', help='')
    parser.add_argument('--dropout', type=float, default=0.3, help='only used for wiede resnet')
    parser.add_argument('--useCifarResNet', type=str2bool, default='False', help='only used for wiede resnet')
    parser.add_argument('--roberta_threshold', type=str, default=None,
                        help='one of 05_None; 05_04; 04_None; 06_None; roberta_superclass20; roberta_superclass40')
    parser.add_argument('--roberta_float_threshold', type=float, default=None, help='')
    parser.add_argument('--coarse_labels', type=str2bool, default='False')
    parser.add_argument('--cifar_superclass_noise_lvl', type=int, default=0)
    parser.add_argument('--k_roberta_sims', type=str2bool, default='False')
    parser.add_argument('--full_imagenet', type=str2bool, default='False')

    # distributed data parallel stuff
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--ngpus_per_node', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='./proc', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--debug', default=False, type=bool,
                        help='debug flag')
    parser.add_argument('--loss_type', default=None, type=str, help='loss type', choices=['rince', 'supcon'])

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # the path according to the environment set
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}_mit_{}_mat_{}_thr{}_memSize_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.trial, opt.min_tau, opt.max_tau,
               opt.similarity_threshold, opt.memorybank_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if not (opt.do_sum_in_log):
        opt.model_name = opt.model_name + 'log_out'
    if opt.mixed_out_in_log:
        opt.model_name = opt.model_name + 'mixed_log_out_in'

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

    opt.ngpus_per_node = torch.cuda.device_count()
    opt.world_size = opt.ngpus_per_node * opt.world_size

    return opt


def set_loader(opt, epoch=0):
    # construct data loader
    # imagenet
    if opt.dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=opt.data_folder, transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)
    print("Dataset size:", len(train_dataset))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_sampler.set_epoch(epoch)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    opt.class_to_idx = train_dataset.class_to_idx

    return train_loader, opt


def set_model(opt, gpu):
    print(f'start run rank {gpu}')
    epoch = 1
    modelType = models.__dict__[opt.model]
    criterion = ContrastiveRanking(opt, modelType)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.num_workers = int((opt.num_workers + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
        criterion = criterion.to(gpu)
        criterion = torch.nn.parallel.DistributedDataParallel(criterion, device_ids=[gpu])
    else:
        raise Exception('no cuda device detected!')

    cudnn.benchmark = True
    print(f'done setup {gpu}')

    return criterion, epoch


def train(train_loader, criterion, optimizer, epoch, opt):
    """one epoch training"""
    criterion.train()
    multiGPU = opt.ngpus_per_node > 1

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images_q = images[0].cuda(opt.gpu, non_blocking=True)
        images_k = images[1].cuda(opt.gpu, non_blocking=True)
        labels = labels.cuda(opt.gpu, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        loss = criterion(images_q, images_k, labels)
        loss = torch.mean(loss)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        if multiGPU:
            torch.cuda.synchronize()
        losses.update(loss.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 and opt.gpu == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def main():
    print('start date: ' + str(time.asctime()))
    opt = parse_option()
    port = find_free_port()
    import socket
    opt.dist_url = 'tcp://' + str(socket.gethostbyname(socket.gethostname())) + ':' + str(port)
    mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt.ngpus_per_node, opt))


def setup(rank, world_size, opt):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(opt.dist_backend, rank=rank, world_size=world_size, init_method=opt.dist_url)


def main_worker(gpu, nprocs=None, opt=None):
    print(f'start main work rank {gpu}')
    opt.gpu = gpu
    torch.cuda.set_device(opt.gpu)

    # setup distributed training
    setup(gpu, opt.world_size, opt)

    # only to get the class_to_index
    _, opt = set_loader(opt)

    # build model and criterion
    criterion, epoch = set_model(opt, gpu)

    train_loader, opt = set_loader(opt, epoch)

    # build optimizer
    optimizer = set_optimizer(opt, criterion.module.backbone_q)

    # resume form checkpoint if available
    start_epoch = 1
    if opt.resume:
        ckpt = torch.load(opt.resume, map_location='cpu')
        criterion.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1

    # tensorboard
    tb_writer = SummaryWriter(log_dir=opt.tb_folder)

    # training
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, criterion, optimizer, epoch, opt)

        time2 = time.time()
        print('pre tensorboard epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # log and save model only at first node to avoide unnecessary writing
        if opt.gpu == 0:
            # tensorboard logger
            tb_writer.add_scalar('train/loss', loss, epoch)
            tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(criterion, optimizer, opt, epoch, save_file)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

    if opt.gpu == 0:
        # save the last model
        if opt.gpu % opt.ngpus_per_node == 0:
            save_file = os.path.join(
                opt.save_folder, 'last.pth')
            save_model(criterion, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
