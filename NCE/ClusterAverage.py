import torch
import torch.nn as nn
import pdb
import functools
import operator

from utils import l2_normalize
from NCE.NCEAverage import NCEAverage


class ClusterAverage(nn.Module):
    def __init__(self, inputSize, outputSize, loader_length, clips_num, T, batch_size, Z_momentum=0.9, memory_momentum=0.5, NN_num=8):
        super(ClusterAverage, self).__init__()
        self.nLen = outputSize  # ndata
        self.embed_dim = inputSize  # embeddings dim
        self.clips_num = clips_num
        self.NN_num = NN_num
        self.loader_length = loader_length

        # Z, T, Z_momentum
        self.register_buffer('params', torch.tensor([-1, T, Z_momentum, memory_momentum]))
        print('[ClusterAverage]: params Z {}, T {}, Z_momentum {}, memory_momentum {}'.format(-1, T, Z_momentum, memory_momentum))

        self.prepare_indices(batch_size, batch_size * self.clips_num)

    # call this method outside
    def init_memoryBank(self, average):
        # using NCEAverage's memory bank to initialize the new one
        self.register_buffer('memory', average.memory.clone())
        print('[ClusterAverage]: initialize memory bank with NCEs memory...')
        print('[ClusterAverage]: size {}, dim {}'.format(self.memory.size(0), self.memory.size(1)))
        print('\tNCEAverage memory mean: {}, ClusterAverage memory mean: {}'.format(average.memory.mean(), self.memory.mean()))

    def prepare_indices(self, batch_size, bs):
        self.batch_size = batch_size
        self.bs = bs
        self.neg_indices = torch.zeros(self.bs, (self.batch_size-1) * self.clips_num).cuda()
        indices_temp = []
        for i in range(self.batch_size):
            indices_temp.append([i + ii * self.batch_size for ii in range(self.clips_num)])
        for i in range(self.bs):
            neg_temp = indices_temp.copy()
            neg_temp.pop(i % self.batch_size)
            self.neg_indices[i, :] = torch.tensor(functools.reduce(operator.iconcat, neg_temp, [])).cuda()  # ((batch_size - 1) * clips_num, )

        self.neg_indices = self.neg_indices.long()

    def get_NN(self):
        similarity = torch.mm(self.embeddings, self.memory.t())
        yd, yi = similarity.topk(self.NN_num, dim=1, largest=True, sorted=True)
        self.yi = yi  # (bs, NN_num)
        # now yi is (bs, NN_num)
        NNs = self.memory[yi]  # (bs, NN_num, dim)

        return NNs

    def compute_data_prob(self):
        NNs = self.get_NN()

        prods = self.embeddings.unsqueeze(1) * NNs  # (bs, 1, dim) * (bs, NN_num, dim)
        logits = torch.sum(prods, dim=-1)  # outcome of innerproduct -- (bs, NN_num)
        logits = torch.mean(logits, dim=-1, keepdim=True)  # (bs, 1)

        return logits

    def compute_noise_prob(self):
        # neg_size = (batch_size-1) * clips_num
        negatives = self.embeddings[self.neg_indices]  # (bs, neg_size, dim)
        prods = self.embeddings.unsqueeze(1) * negatives  # (bs, 1, dim) * (bs, neg_size, dim)
        logits = torch.sum(prods, dim=-1)  # outcome of innerproduct -- (bs, neg_size)

        return logits

    def nce_core(self, pos_logits, neg_logits):
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        outs = torch.exp(logits / self.params[1].item())
        Z = self.params[0].item()
        if Z < 0:
            # initialize it with mean of first batch
            self.params[0] = outs.mean() * self.nLen
            Z = self.params[0].clone().detach().item()
            print('normalization constant Z is set to {:.1f}'.format(Z))
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
        # pdb.set_trace()
        # use nearest neighbour's mean to update memory bank
        new_data_memory = torch.mean(self.memory[self.yi], dim=1)
        new_data_memory = new_data_memory.view(self.clips_num, self.batch_size, self.embed_dim).permute(1, 0, 2)
        # use each clip's neighbours' mean to update
        new_data_memory = torch.mean(new_data_memory, dim=1)
        new_data_memory = data_memory * self.params[3] + (1 - self.params[3]) * new_data_memory
        new_data_memory = l2_normalize(new_data_memory)

        idxs = idxs.unsqueeze(1).repeat(1, self.embed_dim)
        self.memory.scatter_(0, idxs, new_data_memory)

    def forward(self, x, idxs, i):
        bs = x.size(0)
        batch_size = bs // self.clips_num
        self.embeddings = x
        # treat the last batch specially
        if i == self.loader_length - 1:
            self.prepare_indices(batch_size, bs)

        pos_logits = self.compute_data_prob()
        neg_logits = self.compute_noise_prob()
        outs, probs = self.nce_core(pos_logits, neg_logits)
        # pdb.set_trace()

        with torch.no_grad():
            self.update_new_data_memory(idxs)

        return outs, probs


if __name__ == '__main__':
    c_average = ClusterAverage(128, 9000, 239, 4, 0.07, 3).cuda()
    print(c_average.neg_indices)
    n_average = NCEAverage(128, 9000, 239, 4, 0.07, 3).cuda()

    # initialize c_average with n_average
    c_average.init_memoryBank(n_average)

    dummy_embeddings = torch.randn(12, 128).cuda()
    dummy_embeddings = l2_normalize(dummy_embeddings)
    idxs = torch.arange(3).cuda()
    outs_c, probs_c = c_average(dummy_embeddings, idxs, 0)
    outs_n, probs_n, _ = n_average(dummy_embeddings, idxs, 0)
    pdb.set_trace()
