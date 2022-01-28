from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import load_ImageNet_hierarchy


@torch.no_grad()
def gather_all(x):
    tensors_gather = [torch.ones_like(x) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, x, async_op=False)
    return torch.cat(tensors_gather, dim=0)


class ContrastiveRanking(nn.Module):
    def __init__(self, opt, gen_model):
        super(ContrastiveRanking, self).__init__()
        self.multiGPU = opt.ngpus_per_node > 1
        self.gpu = opt.gpu
        self.m = opt.m
        self.supervised_mode = opt.supervised_mode
        self.do_sum_in_log = opt.do_sum_in_log
        self.feature_size = 128

        if opt.model == 'resnet18':
            dim_mlp = 512
        elif opt.model == 'resnet50':
            dim_mlp = 2048

        # create netrworks
        self.backbone_q = gen_model(num_classes=self.feature_size)
        self.backbone_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone_q.fc)
        self.backbone_k = gen_model(num_classes=self.feature_size)
        self.backbone_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone_k.fc)
        for param_k, param_q in zip(self.backbone_k.parameters(), self.backbone_q.parameters()):
            param_q.requires_grad = True
            param_k.data = param_q.data
            param_k.requires_grad = False

        # initalize memorybank
        self.register_buffer("memorybank_InfoNCE", torch.randn(opt.memorybank_size, self.feature_size))
        self.memorybank_InfoNCE = nn.functional.normalize(self.memorybank_InfoNCE, dim=1)
        self.register_buffer("memorybank_labels", torch.ones(opt.memorybank_size, dtype=torch.long) * -1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.min_tau = opt.min_tau
        self.max_tau = opt.max_tau
        self.similarity_threshold = opt.similarity_threshold
        self.sample_n_simclasses = opt.sample_n_simclasses
        self.use_dynamic_tau = opt.use_dynamic_tau
        self.mixed_out_in = opt.mixed_out_in
        self.roberta_threshold = opt.roberta_threshold
        self.roberta_float_threshold = opt.roberta_float_threshold
        self.roberta_threshold = opt.roberta_threshold
        self.cifarload_ImageNet_hierarchy_superclass_noise_lvl = opt.cifar_superclass_noise_lvl
        self.k_roberta_sims = opt.k_roberta_sims

        self.n_sim_classes, self.class_sims_idx = load_ImageNet_hierarchy(
            opt.hierarchy_k, opt.class_to_idx,
            roberta_float_thresholds=self.roberta_float_threshold,
            loss_type=opt.loss_type,
            nameSimsPath=opt.nameSimPath,
            path_name2wordnet=opt.path_name2wordnet)

        self.criterion = ContrastiveRankingLoss()

    @torch.no_grad()
    def shuffle_gpu_batches(self, x):
        batch_size_per_gpu = x.shape[0]
        samples = gather_all(x)
        batch_size_all = samples.shape[0]
        num_gpus = batch_size_all // batch_size_per_gpu
        shuffle_indices = torch.randperm(batch_size_all).to(self.gpu)
        torch.distributed.broadcast(shuffle_indices, src=0)
        unshuffle_indices = torch.argsort(shuffle_indices)

        gpu_idx = torch.distributed.get_rank()
        idx_gpu = shuffle_indices.view(num_gpus, -1)[gpu_idx]
        samples_gpu = samples[idx_gpu]

        return samples_gpu, unshuffle_indices

    @torch.no_grad()
    def unshuffle_gpu_batches(self, x, unshuffle_indices):
        batch_size_per_gpu = x.shape[0]
        samples = gather_all(x)
        batch_size_all = samples.shape[0]
        num_gpus = batch_size_all // batch_size_per_gpu

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_gpu = unshuffle_indices.view(num_gpus, -1)[gpu_idx]
        samples_gpu = samples[idx_gpu]
        return samples_gpu

    def forward(self, images_q, images_k, labels):

        anchor = self.backbone_q(images_q)
        anchor = nn.functional.normalize(anchor, dim=1)
        with torch.no_grad():
            self.update_weights()
            # shuffle for making use of BN
            pos = images_k
            if self.multiGPU:
                pos, idx_unshuffle = self.shuffle_gpu_batches(pos)

            pos = self.backbone_k(pos)
            pos = nn.functional.normalize(pos, dim=1)
            if self.multiGPU:
                pos = self.unshuffle_gpu_batches(pos, idx_unshuffle)

        l_pos, l_class_pos, l_neg, masks, dynamic_taus, pos_enq, labels_enq = self.compute_InfoNCE_classSimilarity(
            anchor=anchor, pos=pos, labels=labels)

        # initially l_neg and l_class pos are identical
        res = {}
        for i, mask in enumerate(masks):
            # mask out from negatives only if they are part of the class and this class has a similarity to
            # label class above the similarity threshold
            l_neg[mask] = -float("inf")
            l_class_pos_cur = l_class_pos.clone()
            # keep only members of current class
            l_class_pos_cur[~mask] = -float("inf")
            taus = dynamic_taus[i].view(-1, 1)

            if i == 0:
                l_class_pos_cur = torch.cat([l_pos, l_class_pos_cur], dim=1)

            if self.mixed_out_in and i == 0:
                loss = self.sum_out_log(l_class_pos_cur, l_neg, taus)
            elif self.do_sum_in_log and not (self.mixed_out_in and i == 0):
                loss = self.sum_in_log(l_class_pos_cur, l_neg, taus)
            else:
                loss = self.sum_out_log(l_class_pos_cur, l_neg, taus)

            result = {'score': None,
                      'target': None,
                      'loss': loss}
            res['class_similarity_ranking_class' + str(i)] = result

        return self.criterion(res, labels)

    def sum_in_log(self, l_pos, l_neg, tau):
        logits = torch.cat([l_pos, l_neg], dim=1) / tau
        logits = F.softmax(logits, dim=1)
        sum_pos = logits[:, 0:l_pos.shape[1]].sum(1)
        sum_pos = sum_pos[sum_pos > 1e-7]
        if len(sum_pos) > 0:
            loss = - torch.log(sum_pos).mean()
        else:
            loss = torch.tensor([0.0]).to(l_pos.device)
        return loss

    def sum_out_log(self, l_pos, l_neg, tau):
        l_pos = l_pos / tau
        l_neg = l_neg / tau
        l_pos_exp = torch.exp(l_pos)
        l_neg_exp_sum = torch.exp(l_neg).sum(dim=1).unsqueeze(1)
        all_scores = (l_pos_exp / (l_pos_exp + l_neg_exp_sum))
        all_scores = all_scores[all_scores > 1e-7]
        if len(all_scores) > 0:
            loss = - torch.log(all_scores).mean()
        else:
            loss = torch.tensor([0.0]).to(l_pos.device)
        return loss

    @torch.no_grad()
    def get_similar_labels(self, labels):

        labels = labels.cpu().numpy()

        sim_class_labels = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).to(self.gpu, torch.long)
        sim_class_sims = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).to(self.gpu, torch.float)
        sim_leq_thresh = torch.zeros(
            (labels.shape[0], len(self.class_sims_idx[0]['sim_class_idx2indices']))).to(self.gpu, torch.bool)
        for i, label in enumerate(labels):
            sim_class_labels[i, :] = self.class_sims_idx[label]['sim_class_idx2indices']
            sim_class_sims[i, :] = self.class_sims_idx[label]['sim_class_val']
            sim_leq_thresh[i, :] = self.class_sims_idx[label]['sim_class_val'] >= self.similarity_threshold
        # remove columns in which no sample has a similarity  equal to or larger than the selected threshold
        at_least_one_leq_thrsh = torch.sum(sim_leq_thresh, dim=0) > 0
        sim_class_labels = sim_class_labels[:, at_least_one_leq_thrsh]

        sim_class_labels = sim_class_labels[:, :self.n_sim_classes]
        sim_class_sims = sim_class_sims[:, :self.n_sim_classes]
        if self.k_roberta_sims:
            sim_class_sims[:, 0] = 1.
            sim_class_sims[:, 1:] = 0.75
            if self.hierarchy_k > 1:
                raise NotImplementedError('only support hierarchy_k=1')

        return sim_class_labels, sim_class_sims

    # returns scores for instance positives, class positives and filtered negatives
    def compute_InfoNCE_classSimilarity(self, anchor, pos, labels, enqueue=True):
        l_pos = torch.einsum('nc,nc->n', [anchor, pos]).unsqueeze(-1)

        similar_labels, class_sims = self.get_similar_labels(labels)
        similar_labels = similar_labels.to(anchor.device)
        class_sims = class_sims.to(anchor.device)
        masks = []

        # mask defines if same label
        for i in range(similar_labels.shape[1]):
            mask = (self.memorybank_labels[:, None] == similar_labels[None, :, i]).transpose(0, 1)
            masks.append(mask)

        # group together discretized similarity measures
        #mask will now encode whether they are positives and have same rank, i.e. discrete similarity score
        similarity_scores = reversed(class_sims.unique(sorted=True))
        similarity_scores = similarity_scores[similarity_scores > -1]
        new_masks = []
        new_taus = []
        for s in similarity_scores:
            new_taus.append(self.get_dynamic_tau(torch.ones_like(class_sims[:, 0]) * s))
            mask_all_siblings = torch.zeros_like(masks[0], dtype=torch.bool).to(anchor.device)
            for i in range(similar_labels.shape[1]):
                same_score = class_sims[:, i] == s
                if any(same_score):
                    mask_all_siblings[same_score] = mask_all_siblings[same_score] | masks[i][same_score]
            new_masks.append(mask_all_siblings)
        masks = new_masks
        dynamic_taus = new_taus

        l_class_pos = torch.einsum('nc,ck->nk', [anchor, self.memorybank_InfoNCE.transpose(0, 1).clone()])
        l_neg = l_class_pos.clone()

        if self.training and enqueue:
            self.enqueue(pos, labels)

        return l_pos, l_class_pos, l_neg, masks, dynamic_taus, pos, labels

    @torch.no_grad()
    def enqueue(self, keys, labels):
        # gather keys before updating queue
        if self.multiGPU:
            keys = gather_all(keys)
            labels = gather_all(labels)
        m_dim = keys.shape[0]

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > self.memorybank_InfoNCE.shape[0]:
            first_idx = self.memorybank_InfoNCE.shape[0] - ptr
            second_idx = batch_size - first_idx

            self.memorybank_InfoNCE[ptr:, :] = keys[:first_idx]
            self.memorybank_labels[ptr:] = labels[:first_idx]
            self.memorybank_InfoNCE[:second_idx, :] = keys[first_idx:]
            self.memorybank_labels[:second_idx] = labels[first_idx:]
            ptr = second_idx
        else:
            self.memorybank_InfoNCE[ptr:ptr + batch_size, :] = keys
            self.memorybank_labels[ptr:ptr + batch_size] = labels
            ptr = ptr + batch_size

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_weights(self):
        dict = {}
        for name, param in self.backbone_q.named_parameters():
            dict[name] = param
        for name, param_k in self.backbone_k.named_parameters():
            if name in dict:
                param_k.data = self.m * param_k.data + (1 - self.m) * dict[name].data

    @torch.no_grad()
    def get_dynamic_tau(self, similarities):
        dissimilarities = 1 - similarities
        d_taus = self.min_tau + (dissimilarities - 0) / (1 - 0) * (self.max_tau - self.min_tau)

        return d_taus


class ContrastiveRankingLoss:
    def __init__(self):
        self.cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        loss = 0.0
        for key, val in outputs.items():
            if 'loss' in val:
                loss = loss + val['loss']
            else:
                loss = loss + self.cross_entropy(val['score'], val['target'])
        loss = loss / float(len(outputs))
        if len(loss.shape) == 0:
            loss = loss.unsqueeze(0)
        return loss
