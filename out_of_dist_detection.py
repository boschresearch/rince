from __future__ import print_function

import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from main_ce import set_loader
from networks.resnet_big import SupConResNet, LinearClassifier, SupCEResNet
from util import AverageMeter
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
    parser.add_argument('--model_type', type=str, default='contrastive', choices=['contrastive', 'cross_entropy'])
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['cifar10', 'cifar100', 'imagenet', 'tiny_imagenet', 'tiny_imagenet_inliers',
                                 'imagenet_100'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--dataset_outliers', type=str, default='imagenet',
                        choices=['cifar10', 'cifar100', 'imagenet', 'tiny_imagenet', 'tiny_imagenet_outliers', 'AwA2'],
                        help='dataset')
    parser.add_argument('--data_folder_outliers', type=str, default=None, help='path to custom dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--size', type=int, default=64, help='parameter for RandomResizedCrop')
    parser.add_argument('--saveBasePath', type=str, default='./save/results/')
    parser.add_argument('--topk', default=[1, 5, 10, 20], nargs='*', type=int)

    parser.add_argument('--use_ssl_augmentations', type=str2bool, default='False')

    opt = parser.parse_args()

    if opt.dataset == 'cifar100' and opt.size == 64:
        raise Exception('wrong size for cifar')

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

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.model_name = opt.ckpt.split('/')[-2]

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'imagenet_100':
        opt.n_cls = 100
    elif opt.dataset == 'imagenet':
        opt.n_cls = 1000
    elif opt.dataset.lower() == 'tiny_imagenet':
        opt.n_cls = 200
    elif opt.dataset == 'tiny_imagenet_inliers':
        opt.n_cls = 180
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    if 'standard' in opt.model:
        model_type = models.__dict__[opt.model.replace('_standard', '')]
        model = model_type(num_classes=opt.n_cls)
        model.fc = torch.nn.Identity()
    elif opt.model_type == 'contrastive':
        model = SupConResNet(name=opt.model)
    elif opt.model_type == 'cross_entropy':
        model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    else:
        raise ValueError(f"Model type not supported: {opt.model_type}")
    criterion = torch.nn.CrossEntropyLoss()

    if 'standard' in opt.model:
        classifier = LinearClassifier(name=opt.model.replace('_standard', ''), num_classes=opt.n_cls)
    else:
        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'standard' in opt.model:
                    if 'backbone_q' in k:
                        k = k.replace('backbone_q.', '')
                        k = k.replace("module.", "")
                        if not k.startswith('fc.'):
                            new_state_dict[k] = v
                    elif 'model.' in k:
                        k = k.replace('model.', '')
                        if 'layer_blocks.' in k:
                            k = k.replace('layer_blocks.', 'layer')
                        new_state_dict[k] = v
                    else:
                        k = k.replace("module.", "")
                        new_state_dict[k] = v
                else:
                    if 'backbone_q' in k:
                        k = k.replace('backbone_q.', '')
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


def validate(val_loader, model, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    features = []
    classes = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if 'standard' in opt.model:
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


def main():
    opt = parse_option()

    # build data loader of inlier classes
    train_loader, val_loader = set_loader(opt, retrieval=True)
    # build data loader of outlier classes
    mean_dataset = opt.dataset
    opt.dataset = opt.dataset_outliers
    opt.data_folder = opt.data_folder_outliers
    _, val_loader_outlier = set_loader(opt, retrieval=True, overwrite_mean_and_std_dataset=mean_dataset)
    # build model > only the backbone is used!
    model, classifier, criterion = set_model(opt)

    # eval for one epoch to compute features
    print("Loading features ...", end=" ")
    train_features, train_labels = validate(train_loader, model, opt)
    test_features, test_labels = validate(val_loader, model, opt)
    print("Inliers done ...", end=" ")
    test_features_outlier, _ = validate(val_loader_outlier, model, opt)
    print("Outliers done.")

    train_features = train_features.cpu().numpy()
    test_features = test_features.cpu().numpy()
    train_labels = train_labels.cpu().numpy()
    test_features_outlier = test_features_outlier.cpu().numpy()

    features_outlier = np.concatenate([test_features_outlier, test_features], axis=0)
    labels = np.concatenate([np.ones(test_features_outlier.shape[0], ),
                             np.zeros(test_features.shape[0], )], axis=0)
    labels2 = np.concatenate([np.zeros(test_features_outlier.shape[0], ),
                              np.ones(test_features.shape[0], )], axis=0)

    gms = {}
    posteriors = np.zeros((features_outlier.shape[0], len(np.unique(train_labels))))
    mahal_dist = np.zeros((features_outlier.shape[0], len(np.unique(train_labels))))
    for i, label in enumerate(np.unique(train_labels)):
        means = np.mean(train_features[train_labels == label, :], axis=0).reshape((1, -1))
        gms[str(label)] = GaussianMixture(1, random_state=0, means_init=means).fit(
            train_features[train_labels == label, :])  # replace mean and cov with exact values
        posteriors[:, i] = gms[str(label)].score_samples(features_outlier)

    max_score = np.max(posteriors, axis=1)
    max_mahal_score = np.max(mahal_dist, axis=1)

    auroc = roc_auc_score(labels2, max_score)
    print('AUTROC: ' + str(auroc))


if __name__ == '__main__':
    main()
