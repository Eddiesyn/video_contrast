import torch
import torch.nn as nn
import math
import functools
import operator
import pdb
import numpy as np

from utils import l2_normalize
from NCE.alias_multinomial import AliasMethod


class NCEAverage(nn.Module):
    '''NCE Average Module based on https://github.com/zhirongw/lemniscate.pytorch
    This Module is designed for spatial temporal action recognition in unsupervised manner
    where each sample contains multiple clips instead of only one instance, positive pairs
    are each clip with its neighbour clips coming from the same video (intra-batch contrast)
    negative pairs are each clip with all other clips from other videos

    inputSize: dimension of output feature
    outputSize: number of features in memory bank -- number of training samples
    loader_length: total steps of training loader, needed for special treatment of last batch
    clips_num: number of clips in each video
    T: temperature of nce
    batch_size: specified batchsize in loader, each batch contains batchsize * clips_num samples ordering in 0,1,2,3...0,1,2,3...
    Z_momentum: update parameter for normalization factor
    memory_momentum: update parameter for memory bank
    '''
    def __init__(self, inputSize, outputSize, loader_length, clips_num, T, batch_size, Z_momentum=0.9, memory_momentum=0.5):
        super(NCEAverage, self).__init__()
        print("[NCE]: Initialization...")
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.clips_num = clips_num
        self.loader_length = loader_length

        self.register_buffer('params', torch.tensor([-1, T, Z_momentum, memory_momentum]))  # [Z, T, Z_momentum, memory_momentum]
        print("[NCE]: params \n\tZ: {}, \n\tT: {}, \n\tZ_momentum {}, \n\tmemory_momentum {}".format(-1, T, Z_momentum, memory_momentum))

        stdv = 1. / math.sqrt(self.inputSize / 3)
        self.register_buffer('memory', torch.rand(self.outputSize, self.inputSize).mul_(2 * stdv).add_(-stdv))
        print("[NCE]: memory bank, {} {}-dim random unit vectors".format(self.outputSize, self.inputSize))

        # prepare indices for slicing
        self.prepare_indices(batch_size, batch_size*self.clips_num)

    # deprecated
    def remove_self(self, V_out):
        # ======== remove diagonal elements of V_out ========== #
        assert V_out.size(0) == V_out.size(1)

        return V_out[~torch.eye(V_out.shape[0], dtype=torch.bool)].view(V_out.shape[0], -1)

    # deprecated
    def reorder_out(self, V_out):
        # ======== move positive pair to the first column ======= #
        assert V_out.size(0) == V_out.size(1)
        positive_pairs = torch.diag(V_out).unsqueeze(1)
        negative_pairs = self.remove_self(V_out)

        return torch.cat([positive_pairs, negative_pairs], dim=1)

    def prepare_indices(self, batch_size, bs):
        '''this func prepare pos_idx and neg_idx for each sample clip
        each sample clip has clips_num-1 positive pairs and (batch_size-1)*clips_num negative pairs
        this func is done once in initialization, and needs to be invoked again for the last batch
        since the batch_size will be different

        batch_size: specified batchsize
        bs: real size of each batch -- batch_size * clips_num
        '''
        self.batch_size = batch_size
        self.bs = bs
        self.indices = torch.arange(self.bs).view(self.clips_num, self.batch_size).t().cuda()  # this is for computing memory update
        self.pos_indices = torch.zeros(self.bs, self.clips_num-1).cuda()
        self.neg_indices = torch.zeros(self.bs, (self.batch_size-1) * self.clips_num).cuda()
        indices_temp = []
        for i in range(self.batch_size):
            indices_temp.append([i + ii * self.batch_size for ii in range(self.clips_num)])
        for i in range(self.bs):
            pos_temp = indices_temp[i % self.batch_size].copy()
            pos_temp.remove(i)
            neg_temp = indices_temp.copy()
            neg_temp.pop(i % self.batch_size)
            self.pos_indices[i, :] = torch.tensor(pos_temp).cuda()  # (clips_num-1)
            self.neg_indices[i, :] = torch.tensor(functools.reduce(operator.iconcat, neg_temp, [])).cuda()  # ((batch_size-1)*clips_num, )

        self.pos_indices = self.pos_indices.long()
        self.neg_indices = self.neg_indices.long()

    def compute_data_prob(self):
        '''method for computing cosine similarity between positive pairs'''

        # (bs, clips_num-1, embed_dim): each feature and its clips_num-1 neighbour clips from same video
        positives = self.embeddings[self.pos_indices]
        # (bs, clips_num-1, dim) @ (bs, dim, 1) = (bs, clips_num-1)
        logits = torch.bmm(positives, self.embeddings.view(self.embeddings.size(0), self.embeddings.size(1), 1)).squeeze()
        # use mean similarity
        logits = torch.mean(logits, dim=1, keepdim=True)  # (bs, 1)

        return logits

    def compute_noise_prob(self):
        '''method for computing cosine similarity between negative pairs'''

        # neg_size = (batch_size-1) * clips_num
        negatives = self.embeddings[self.neg_indices]  # (bs, neg_size, embed_dim)
        # (bs, neg_size, dim) @ (bs, dim, 1) = (bs, neg_size)
        logits = torch.bmm(negatives, self.embeddings.view(self.embeddings.size(0), self.embeddings.size(1), 1)).squeeze()

        return logits

    def nce_core(self, pos_logits, neg_logits):
        '''noise contrastive estimation'''
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        outs = torch.exp(logits / self.params[1].item())
        Z = self.params[0].item()
        if Z < 0:
            # initialize it with mean of first batch
            self.params[0] = outs.mean() * self.outputSize
            Z = self.params[0].clone().detach().item()
            print('normalization constant Z is set to {:.1f}'.format(Z))
        else:
            Z_new = outs.mean() * self.outputSize

            # former implementation wrong !!! (delete after experiment)
            # self.params[0] = self.params[2] * Z_new + (1 - self.params[2]) * self.params[0]

            # update normalization factor with self.params[2]
            self.params[0] = (1 - self.params[2]) * Z_new + self.params[2] * self.params[0]
            Z = self.params[0].clone().detach().item()

        outs = torch.div(outs, Z).contiguous()
        probs = self.extract_probs(outs)

        return outs, probs

    def extract_probs(self, out):
        probs = out / torch.sum(out, dim=1, keepdim=True)

        return probs[:, 0].mean()

    def update_new_data_memory(self, idxs):
        '''update memory bank with mean feature of clips from each video'''
        data_memory = torch.index_select(self.memory, 0, idxs)
        # pdb.set_trace()
        new_data_memory = torch.mean(self.embeddings[self.indices], dim=1)
        new_data_memory = data_memory * self.params[3] + (1 - self.params[3]) * new_data_memory
        new_data_memory = l2_normalize(new_data_memory)

        idxs = idxs.unsqueeze(1).repeat(1, self.inputSize)
        self.memory.scatter_(0, idxs, new_data_memory)

    def recompute_memory(self, idxs):
        new_data_memory = torch.mean(self.embeddings[self.indices], dim=1)
        new_data_memory = l2_normalize(new_data_memory)
        idxs = idxs.unsqueeze(1).repeat(1, self.inputSize)
        self.memory.scatter_(0, idxs, new_data_memory)

    def forward(self, x, idxs, i):
        '''Only use intra batch noise samples, memory bank is only for test(kNN)
        x: embeddings (bs, embed_dim) where bs = args.batch_size * clips_num
        idxs: index of each sample video
        '''

        bs = x.size(0)
        batch_size = bs // self.clips_num
        self.embeddings = x

        # treat the last batch specially
        if i == self.loader_length - 1:
            self.prepare_indices(batch_size, bs)

        pos_logits = self.compute_data_prob()
        neg_logits = self.compute_noise_prob()
        outs, normed_probs = self.nce_core(pos_logits, neg_logits)

        with torch.no_grad():
            self.update_new_data_memory(idxs)
            # self.recompute_memory(idxs)

        return outs, normed_probs


class MemoryInsDis(nn.Module):
    """Memory bank style NCE average"""
    def __init__(self, inputSize, outputSize, loader_length, clips_num, batch_size, K, average,
                 T=0.07, Z_momentum=0.9, memory_momentum=0.5):
        # average is the NCEAverage instance to be inherited from
        super(MemoryInsDis, self).__init__()
        self.nLen = outputSize
        self.embed_dim = inputSize
        self.clips_num = clips_num
        self.loader_length = loader_length
        self.K = K

        self.register_buffer('params', average.params.clone())
        print("[NCE]: Use params from NCEAverage...")
        print("[NCE]: Z {}, T {}, Z_momentum {}, memory_momentum {}".format(self.params[0],
                                                                            self.params[1],
                                                                            self.params[2],
                                                                            self.params[3]))
        self.register_buffer('memory', average.memory.clone())
        print('[NCE]: Use memory bank from NCEAverage...')
        print('[NCE]: {} {}-dim, mean: {}'.format(self.memory.size(0), self.memory.size(1), self.memory.mean()))

        # self.register_buffer('params', torch.tensor([-1, T, K, Z_momentum, memory_momentum]))
        # print("[NCE]: params Z {}, T {}, K {}, Z_momentum {}, memory_momentum {}".format(-1, T, K, Z_momentum, memory_momentum))
        # stdv = 1. / math.sqrt(self.embed_dim / 3)
        # self.register_buffer('memory', torch.rand(self.nLen, self.embed_dim).mul_(2 * stdv).add_(-stdv))
        # print("[NCE]: memory bank, {} {}-dim random unit vectors".format(self.nLen, self.embed_dim))

        self.unigrams = torch.ones(self.nLen)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()

        self.prepare_indices(batch_size, batch_size * self.clips_num)

    def prepare_indices(self, batch_size, bs):
        '''this func prepares pos_idx for each sample clip
        '''
        self.indices = torch.arange(bs).view(self.clips_num, batch_size).t().cuda()  # this is for memory update
        self.pos_indices = torch.zeros(bs, self.clips_num-1).cuda()
        indices_temp = []
        for i in range(batch_size):
            indices_temp.append([i + ii * batch_size for ii in range(self.clips_num)])
        for i in range(bs):
            pos_temp = indices_temp[i % batch_size].copy()
            pos_temp.remove(i)
            self.pos_indices[i, :] = torch.tensor(pos_temp).cuda()  # (clips_num-1)

        self.pos_indices = self.pos_indices.long()

    def compute_data_prob(self):
        positives = self.embeddings[self.pos_indices]  # (bs, clips_num-1, embed_dim)
        prods = self.embeddings.unsqueeze(1) * positives  # (bs, 1, embed_dim) * (bs, clips_num-1, embed_dim)
        logits = torch.sum(prods, dim=-1)  # outcome of innerproduct between features - (bs, clips_num-1)
        logits = torch.mean(logits, dim=-1, keepdim=True)  # (bs, 1)

        return logits

    def compute_noise_prob(self, bs):
        noise_idx = self.multinomial.draw(bs * self.K).view(bs, -1)  # each clip has K noise samples
        noise_weight = torch.index_select(self.memory, 0, noise_idx.view(-1))
        noise_weight = noise_weight.view(bs, self.K, self.embed_dim)  # (bs, K, dim)
        logits = torch.bmm(noise_weight, self.embeddings.unsqueeze(-1))  # (bs, K, dim) @ (bs, dim, 1)

        return logits  # (bs, K, 1)

    def nce_core(self, pos_logits, neg_logits):
        logits = torch.cat([pos_logits, neg_logits.squeeze(-1)], dim=-1)  # (bs, K+1, 1)
        outs = torch.exp(logits / self.params[1].item())
        Z = self.params[0].item()
        if Z < 0:
            # initialize it with first batch
            self.params[0] = outs.mean() * self.nLen
            Z = self.params[0].clone().detach().item()
            print('normalization constant Z is set to {}'.format(Z))
        else:
            Z_new = outs.mean() * self.nLen
            self.params[0] = self.params[2] * Z_new + (1 - self.params[2]) * self.params[0]
            Z = self.params[0].clone().detach().item()

        outs = torch.div(outs, Z).contiguous()
        probs = self.extract_probs(outs)

        return outs, probs

    def extract_probs(self, out):
        probs = out / torch.sum(out, dim=1, keepdim=True)

        return probs[:, 0].mean()

    def update_new_data_memory(self, idxs):
        data_memory = torch.index_select(self.memory, 0, idxs)
        new_data_memory = torch.mean(self.embeddings[self.indices], dim=1)
        new_data_memory = data_memory * self.params[3] + (1 - self.params[3]) * new_data_memory
        new_data_memory = l2_normalize(new_data_memory)

        idxs = idxs.unsqueeze(1).repeat(1, self.embed_dim)
        self.memory.scatter_(0, idxs, new_data_memory)

    def forward(self, x, idxs, i):
        '''draw noise samples from memory bank so to allow more noise samples'''
        bs = x.size(0)
        batch_size = bs // self.clips_num
        self.embeddings = x
        if i == self.loader_length - 1:
            self.prepare_indices(batch_size, bs)
        # pdb.set_trace()
        pos_logits = self.compute_data_prob()
        neg_logits = self.compute_noise_prob(bs)
        outs, probs = self.nce_core(pos_logits, neg_logits)

        with torch.no_grad():
            self.update_new_data_memory(idxs)

        return outs, probs


class Average(nn.Module):
    """This is for supervised contrastive training"""
    def __init__(self, inputSize, outputSize, sample_num, class_num, T, Z_momentum=0.9, memory_momentum=0.5):
        super(Average, self).__init__()
        print('[NCE]: Initialization...')
        self.outputSize = outputSize  # ndata
        self.inputSize = inputSize  # low_dim
        self.sample_num = sample_num
        self.class_num = class_num

        # keep it consistent as pretraining so we could load it
        self.register_buffer('params', torch.tensor([-1, T, Z_momentum, memory_momentum]))
        stdv = 1. / math.sqrt(self.inputSize / 3)
        self.register_buffer('memory', torch.rand(self.outputSize, self.inputSize).mul_(2*stdv).add_(-stdv))
        print('[NCE]: memory bank, {} {}-dim random unit vectors'.format(self.outputSize, self.inputSize))

        # prepare indices for slicing
        self.prepare_indices()

    def prepare_indices(self):
        indices_temp = np.arange(self.sample_num * self.class_num).reshape((self.class_num, self.sample_num)).tolist()
        self.pos_indices = torch.zeros(self.sample_num * self.class_num, self.sample_num-1).cuda()
        self.neg_indices = torch.zeros(self.sample_num * self.class_num, (self.class_num-1)*self.sample_num).cuda()
        for i in range(self.sample_num * self.class_num):
            pos_temp = indices_temp[i // self.sample_num].copy()
            pos_temp.remove(i)
            neg_temp = indices_temp.copy()
            neg_temp.pop(i // self.sample_num)

            self.pos_indices[i, :] = torch.tensor(pos_temp).cuda()  # (sample_num-1, )
            # ((class_num-1)*sample_num, )
            self.neg_indices[i, :] = torch.tensor(functools.reduce(operator.iconcat, neg_temp, [])).cuda()
        self.pos_indices = self.pos_indices.long()
        self.neg_indices = self.neg_indices.long()

    def compute_data_prob(self):
        # pdb.set_trace()
        positives = self.embeddings[self.pos_indices]
        # (160, 1, 512) @ (160, 512, 1) = (160, 1, 1)
        logits = torch.bmm(positives, self.embeddings.view(self.embeddings.size(0), self.embeddings.size(1), 1)).squeeze(-1)

        return logits

    def compute_noise_prob(self):
        # pdb.set_trace()
        negatives = self.embeddings[self.neg_indices]
        # (160, 158, 512) @ (160, 512, 1) = (160, 158, 1)
        logits = torch.bmm(negatives, self.embeddings.view(self.embeddings.size(0), self.embeddings.size(1), 1)).squeeze(-1)

        return logits

    def extract_probs(self, out):
        probs = out / torch.sum(out, dim=1, keepdim=True)

        return probs[:, 0].mean()

    def nce_core(self, pos_logits, neg_logits):
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        outs = torch.exp(logits / self.params[1].item())  # divide the temperature
        Z = self.params[0].item()
        if Z < 0:
            self.params[0] = outs.mean() * self.outputSize
            Z = self.params[0].clone().detach().item()
            print('normalization constant Z is set to {:.1f}'.format(Z))
        else:
            Z_new = outs.mean() * self.outputSize
            self.params[0] = (1 - self.params[2]) * Z_new + self.params[2] * self.params[0]
            Z = self.params[0].clone().detach().item()

        outs = torch.div(outs, Z).contiguous()
        probs = self.extract_probs(outs)

        return outs, probs

    def update_new_data_memory(self, idxs):
        data_memory = torch.index_select(self.memory, 0, idxs)
        new_data_memory = data_memory * self.params[3] + (1 - self.params[3]) * self.embeddings
        new_data_memory = l2_normalize(new_data_memory)

        idxs = idxs.unsqueeze(1).repeat(1, self.inputSize)
        self.memory.scatter_(0, idxs, new_data_memory)

    def forward(self, x, idxs):
        self.embeddings = x

        pos_logits = self.compute_data_prob()
        neg_logits = self.compute_noise_prob()
        # pdb.set_trace()
        outs, normed_probs = self.nce_core(pos_logits, neg_logits)

        with torch.no_grad():
            self.update_new_data_memory(idxs)

        return outs, normed_probs


if __name__ == '__main__':
    average = NCEAverage(128, 9000, 1000, 4, 0.07, 3).cuda()
    print(average.pos_indices)
    print(average.neg_indices)
    dummy_embeddings = torch.randn(12, 128).cuda()
    dummy_embeddings = l2_normalize(dummy_embeddings)
    idxs = torch.arange(3).cuda()
    outs, probs = average(dummy_embeddings, idxs, 0)
    # print(outs)
    # pdb.set_trace()
    print(outs.shape)
    # pdb.set_trace()
