import csv
import torch
from torch.utils.data.sampler import Sampler
import itertools
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import os


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


class Logger(object):
    """Logger object for training process, supporting resume training"""
    def __init__(self, path, header, resume=False):
        """Constructor

        :param path: logging file path
        :param header: a list of tags for values to track
        :param resume: a flag controling whether to create a new
        file or continue recording after the latest step
        """
        self.log_file = None
        self.resume = resume
        self.header = header
        if not self.resume:
            self.log_file = open(path, 'w')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(self.header)
        else:
            self.log_file = open(path, 'a+')
            self.log_file.seek(0, os.SEEK_SET)
            reader = csv.reader(self.log_file, delimiter='\t')
            self.header = next(reader)
            # move back to the end of file
            self.log_file.seek(0, os.SEEK_END)
            self.logger = csv.writer(self.log_file, delimiter='\t')

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for tag in self.header:
            assert tag in values, 'Please give the right value as defined'
            write_values.append(values[tag])
        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, opt):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name), '%s/%s_best.pth' % (opt.result_path, opt.store_name))


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new


def pdist_torch(embeddings):
    '''
    compute the eucilidean distance matrix between different embeddings
    --embeddings: size (n, d) features
    '''
    n = embeddings.size(0)
    dist = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, embeddings, embeddings.t())
    dist = dist.clamp(min=1e-12).sqrt()

    return dist


def pearson_coefficient(embedding):
    '''
    calculate pearson coefficient between different samples' embeddings
    --input embedding: shape (n_samples, n_features)
    --result matrix: shape (n_samples, n_samples)
    '''
    mean_x = torch.mean(embedding, 1)
    xm = embedding - mean_x.unsqueeze(0).t()
    c = xm.mm(xm.t())
    c = c / (embedding.size(1) - 1)

    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, 0.0, 1.0)

    # based on paper "Group loss for deep metric learning", dignonal entries are set to 0
    c.fill_diagonal_(0.)

    return c


def pearson_coefficient_bank(embedding, memory_bank):
    """calculate pearson correlation between embeddings and correspondings in memory bank"""
    e_c = embedding - torch.mean(embedding, dim=1, keepdim=True)
    m_c = memory_bank - torch.mean(memory_bank, dim=1, keepdim=True)

    up = torch.mm(e_c, m_c.t())
    res = up / torch.sqrt(torch.sum(e_c**2, dim=1, keepdim=True).mm(torch.sum(m_c**2, dim=1, keepdim=True).t()))

    return torch.clamp(res, -1.0, 1.0)


def update(W, X):
    pai = torch.mm(W, X)
    nominator = torch.mul(X, pai)
    q = torch.sum(nominator, dim=1)
    q_inv = torch.diag(1 / q)
    new_x = q_inv.mm(nominator)

    return new_x


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))


def show_distribution(yd, NN_labels, labels):
    '''plot similarity distribution of postives and negatives

    yd: similarity of nearest neighbours(sorted) -- shape (bs, K)
    NN_labels: labels of nearest neighbours -- shape (bs, K)
    labels: true labels -- shape (bs, )
    '''
    assert yd.size(0) == NN_labels.size(0) == labels.size(0), 'Fatal ERROR!'
    bs = yd.size(0)
    pos_similarity = []
    neg_similarity = []

    postives = NN_labels.eq(labels.view(-1, 1))
    for i in range(bs):
        pos_idx = torch.nonzero(postives[i, :]).view(-1)
        neg_idx = torch.nonzero(~postives[i, :]).view(-1)

        pos_similarity.extend(yd[i, pos_idx].tolist())
        neg_similarity.extend(yd[i, neg_idx].tolist())

    assert len(pos_similarity) + len(neg_similarity) == yd.size(0) * yd.size(1), 'Fatal ERROR!'

    plt.figure()
    plt.hist(pos_similarity, bins=100, label='pos_scores')
    plt.hist(neg_similarity, bins=100, label='neg_scores', alpha=0.5)
    plt.legend(loc='best')
    plt.savefig('./results/cifar/hist.png')


def grouper(iterable, n):
    args = [iter(iterable)] * n

    return zip(*args)


class ContrastiveSampler(Sampler):
    """
    Contrastive batchSampler: extract n_classes * n_samples for each batch
    inspired from: https://github.com/adambielski/siamese-triplet/blob/master/datasets.py

    lookup: a dict containing the labels of specific label
    n_classes: number of different classes within each batch
    n_samples: number of samples for each class
    ndata: number of samples in dataset
    """
    def __init__(self, lookup, n_classes, n_samples, ndata):
        self.lookup = lookup
        self.labels_set = list(self.lookup.keys())
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batchsize = self.n_classes * self.n_samples

        for i in self.lookup.keys():
            random.shuffle(self.lookup[i])
        self.used_label_indices_count = {label: 0 for label in self.lookup.keys()}
        self.count = 0
        self.n_dataset = ndata

    def __iter__(self):
        self.count = 0
        while self.count + self.batchsize < self.n_dataset:
            classes = random.sample(self.labels_set, self.n_classes)
            indices = []
            for class_ in classes:
                indices.extend(self.lookup[class_][self.used_label_indices_count[class_]:
                                                   self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.lookup[class_]):
                    random.shuffle(self.lookup[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_samples * self.n_classes

    def __len__(self):
        return self.n_dataset // self.batchsize


def set_lr(optimizer, lr_rate):
    """set a new learning rate, used in resume training with new learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_rate
