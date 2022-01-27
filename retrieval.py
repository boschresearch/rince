from __future__ import print_function

import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from main_ce import set_loader
from networks.resnet_big import SupConResNet, LinearClassifier, SupCEResNet
from util import AverageMeter
from util import set_optimizer
from util import str2bool

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--model_type', type=str, default='contrastive')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--data_folder', type=str, default="./Data/",
                        help='path to custom dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--saveBasePath', type=str, default='./save/results/')
    parser.add_argument('--topk', default=[1, 5, 10, 20], nargs='*', type=int)
    parser.add_argument('--labelset', type=str, default='fine', choices=['fine', 'coarse', 'both'])
    parser.add_argument('--after_MLP', default=False, type=str2bool)
    parser.add_argument('--run', default=-1, type=int)
    parser.add_argument('--precision_recall', default=True, type=str2bool)
    parser.add_argument('--use_ssl_augmentations', type=str2bool, default='False')
    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    if opt.model_type == 'contrastive':
        model = SupConResNet(name=opt.model)
    elif opt.model_type == 'cross_entropy':
        model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    else:
        raise ValueError(f"Model type not supported: {opt.model_type}")
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("backbone_q.", "")
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        loading_msg = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", loading_msg.missing_keys)

    return model, classifier, criterion


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    features = []
    classes = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if opt.after_MLP:
                output = model(images)
            else:
                output = model.encoder(images)
            features.append(output)
            classes.append(labels)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    features = torch.cat(features)
    features = nn.functional.normalize(features, dim=1)
    labels = torch.cat(classes)

    return features, labels


def retrieve(test_features, train_features, test_labels, train_labels, K, dataset, opt):
    N = len(test_labels)
    sim = torch.mm(test_features, train_features.transpose(0, 1))
    k = max(K)
    _, topk = torch.topk(sim, k)
    acc_mat = torch.zeros((N, k))
    for i in range(k):
        pred = train_labels[topk[:, i]]
        retrieved = torch.tensor(test_labels == pred, dtype=torch.int8)
        acc_mat[:, i] = retrieved  # [:, 0]
    topk_acc = []
    content = ''
    for i in K:
        retr, _ = acc_mat[:, 0:i].max(dim=1)
        acc = retr.float().sum() / float(N)
        topk_acc.append(acc)
        content += f'Recall R@{i}: {acc:.4f} '
    content += '\t'
    print(f'{dataset}:' + content)


def precision_recall(test_features, train_features, test_labels, train_labels):
    n_pos = (test_labels.unsqueeze(1) == train_labels.unsqueeze(0)).sum(1).cpu()
    N = len(test_labels)
    sim = torch.mm(test_features, train_features.transpose(0, 1))
    k = len(train_labels)
    _, topk = torch.topk(sim, k)
    acc_mat = torch.zeros((N, k))
    for i in range(k):
        pred = train_labels[topk[:, i]]
        retrieved = torch.tensor(test_labels == pred, dtype=torch.int8)
        acc_mat[:, i] = retrieved
    precision_recalls = {'precision': [], 'recall': []}

    for i in range(1, k + 1):
        if i % 1000 == 0: print(f"{i} of {k} done")
        n_correct = acc_mat[:, 0:i].sum(1)
        precision = n_correct / i
        recall = n_correct / n_pos
        precision_recalls['precision'].append(precision.mean())
        precision_recalls['recall'].append(recall.mean())

    p = torch.stack(precision_recalls['precision'])
    r = torch.stack(precision_recalls['recall'])
    AP = ((r[1:] - r[:-1]) * p[1:]).sum()
    precision_recalls["AP"] = AP
    print(f"AP: {AP}")

    return precision_recalls


if __name__ == '__main__':
    opt = parse_option()
    opt.model_name = opt.ckpt.split('/')[-2]

    print(f"Retrieve model {opt.model_name}")

    # build data loader
    train_loader, val_loader = set_loader(opt, retrieval=True, labelset=opt.labelset)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier, islinear=True)

    # compute features
    print("Compute features...", end=" ")
    train_features, train_labels = validate(train_loader, model, classifier, criterion, opt)
    test_features, test_labels = validate(val_loader, model, classifier, criterion, opt)
    print("done.")

    topk = retrieve(test_features, train_features, test_labels, train_labels, opt.topk, opt.dataset, opt)

    # compute precision-recall
    if opt.precision_recall:
        print("Compute precision and recall...")
        precision_recall(test_features, train_features, test_labels, train_labels)
