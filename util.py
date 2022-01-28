from __future__ import print_function

import math

import numpy as np
import torch
import torch.optim as optim


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model, islinear=False):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    if hasattr(opt, 'ckpt') and not islinear:
        if not opt.ckpt is None:
            ckpt = torch.load(opt.ckpt, map_location='cpu')
            optimizer.load_state_dict(ckpt['optimizer'])

    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_class_distances_dummy(k):
    class_distances = torch.load("imagenet100_dummyDists.pt")
    vals = []
    for cd in class_distances.keys():
        vals.append(sum(np.array(list(class_distances[cd].values())) <= k))
    n_sim_classes = max(vals)
    return class_distances, n_sim_classes


def get_class_distances_Roberta_threshold(k, float_threshold=None, nameSimsPath=None, path_name2wordnet=None):
    def add_to_dict(word_distances, name2wordnet, sim_word, insertAt, to_add):
        if isinstance(name2wordnet[sim_word], list):
            for el in name2wordnet[sim_word]:
                word_distances[insertAt][el] = to_add
        else:
            word_distances[insertAt][name2wordnet[sim_word]] = to_add

        return word_distances

    nameSims = np.load(nameSimsPath, allow_pickle=True)
    nameSims = nameSims.item()
    name2wordnet = np.load(path_name2wordnet, allow_pickle=True)
    name2wordnet = name2wordnet.item()
    threshold = float_threshold

    word_distances = {}
    for word in nameSims.keys():
        similarities = nameSims[word]['similarities']
        similar_words = nameSims[word]['words']

        if not (isinstance(name2wordnet[word], list)):
            name2wordnet[word] = [name2wordnet[word]]

        for n2w in name2wordnet[word]:
            word_distances[n2w] = {}
            # check if similarity larger threshold. If yes, set distance to 1 else set distance to large value
            for sim_word, sims in zip(similar_words, similarities):
                if sim_word == word:
                    word_distances = add_to_dict(word_distances, name2wordnet, sim_word, n2w, 0)
                elif sims >= threshold:
                    word_distances = add_to_dict(word_distances, name2wordnet, sim_word, n2w, 1)
                else:
                    word_distances = add_to_dict(word_distances, name2wordnet, sim_word, n2w, 100)
    class_distances = word_distances

    vals = []
    for cd in class_distances.keys():
        vals.append(sum(np.array(list(class_distances[cd].values())) <= k))
    n_sim_classes = max(vals)
    return class_distances, n_sim_classes


def load_ImageNet_hierarchy(k, class_to_idx, roberta_float_thresholds=None, loss_type=None, nameSimsPath=None, path_name2wordnet=None):
    if loss_type == 'rince':
        class_distances, n_sim_classes = get_class_distances_Roberta_threshold(k, roberta_float_thresholds, nameSimsPath, path_name2wordnet)
    elif loss_type == 'supcon':
        class_distances, n_sim_classes = load_class_distances_dummy(k)
    else:
        raise Exception('not defined')

    class_sims_idx = {}
    for cls, sim_cls in class_distances.items():
        idx = class_to_idx[cls]
        class_sims_idx[idx] = {}
        sim_idx = []
        similarity = []
        # get class similaritie and respective indices as list for class cls
        for s_cls, cls_dist in sim_cls.items():
            s_idx = class_to_idx[s_cls]
            if cls_dist <= k:
                sim_idx.append(s_idx)
                similarity.append(1.0 - 0.25 * cls_dist)
        # sort both lists accordding to similarity
        sim_idx_similarity = [(cls_, sim_) for cls_, sim_ in zip(sim_idx, similarity)]
        sim_idx_similarity = sorted(sim_idx_similarity, key=lambda x: x[1], reverse=True)
        sim_idx = [x[0] for x in sim_idx_similarity]
        similarity = [x[1] for x in sim_idx_similarity]
        # get vector with indices and -2 at empty position
        idx2indices = -2 * torch.ones(n_sim_classes).type(torch.long)
        idx2indices[0:len(sim_idx)] = torch.tensor(sim_idx).type(torch.long)
        class_sims_idx[idx]['sim_class_idx2indices'] = idx2indices.cuda()
        # get similarity vector
        sim = -1.0 * torch.ones(n_sim_classes)
        sim[0:len(similarity)] = torch.tensor(similarity).type(torch.float)
        class_sims_idx[idx]['sim_class_val'] = sim.cuda()
    return n_sim_classes, class_sims_idx
