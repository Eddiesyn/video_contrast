import torch
import torch.nn as nn
import math


class MemoryMoCo(nn.Module):
    """Fixed-sized queue with momentum encoder based on https://github.com/HobbitLong/CMC
    This Module is designed for spatial temporal action recognition in unsupervised manner
    where each sample with its neighbour clips coming from the same video (intra-batch contrast)
    negative pairs are each clip with all other clips from other videos

    inputSize: dimension of output feature
    outputSize: number of features in trainset
    batch_size: specified batchsize in loader, each batch contains batchsize * clips_num samples
    clips_num:
    """
    def __init__(self, inputSize, outputSize, batch_size, clips_num, loader_length, K, Z_momentum=0.9, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        print('[MoCo]: Initialization...')
        self.outputSize = outputSize  # number of samples
        self.inputSize = inputSize  # embedding dim
        self.batch_size = batch_size
        self.clips_num = clips_num
        self.loader_length = loader_length
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        # Z and its update momentum is as params
        self.register_buffer('params', torch.tensor([-1, Z_momentum]))
        print('[MoCo]: params Z {}, Z_momentum {}'.format(self.params[0], self.params[1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2*stdv).add_(-stdv))
        print('[MoCo]: Using queue shape: ({}, {})'.format(self.queueSize, inputSize))

        # prepare indices for queue update
        self.prepare_indices(batch_size, batch_size*clips_num)

    def prepare_indices(self, batch_size, bs):
        '''This func provide pos_idx for each clip'''
        self.indices = torch.arange(bs).view(self.clips_num, batch_size).t().cuda()  # this is for update
        self.pos_indices = torch.zeros(bs, self.clips_num-1).cuda()
        indices_temp = []
        for i in range(batch_size):
            indices_temp.append([i + ii * batch_size for ii in range(self.clips_num)])
        for i in range(bs):
            pos_temp = indices_temp[i % batch_size].copy()
            pos_temp.remove(i)
            self.pos_indices[i, :] = torch.tensor(pos_temp).cuda()  # (clips_num-1)

        self.pos_indices = self.pos_indices.long()

    def compute_data_prob(self, q, k):
        positives = k[self.pos_indices]  # (bs, clips_num-1, dim)
        logits = torch.bmm(positives, q.unsqueeze(-1))  # (bs, clips_num-1, dim) @ (bs, dim, 1) = (bs,clips_num,1)
        logits = logits.squeeze(-1)  # (bs, clips_num-1)
        logits = torch.mean(logits, dim=-1, keepdim=True)  # (bs, 1)

        return logits

    def extract_probs(self, out):
        probs = out / torch.sum(out, dim=1, keepdim=True)

        return probs[:, 0].mean()

    def forward(self, q, k, i):
        '''
        q: query, coming from current network (one with BP)
        k: key, coming from momentum updated network (one without BP)
        '''
        bs = q.size(0)
        batchSize = bs // self.clips_num
        k = k.detach()  # no bp for key encoder

        # special treatment for last batch
        if i == self.loader_length - 1:
            self.prepare_indices(batchSize, bs)

        Z = self.params[0].item()


        # ========== original implementation =============
        # positive logtis is contrast between query and key
        # query is transform1(input)'s embedding of query network
        # key is transform2(input)'s embedding of key network
        # negative logits is contrast between query and queue
        # queue is updated by key
        # ================================================
        # l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        # l_pos = l_pos.view(batchSize, 1)
        # queue = self.memory.clone()
        # l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        # l_neg = l_neg.transpose(0, 1)

        # =========== current implementation ===============
        # positive logits is contrast between different clip's embedding of same input (query network)
        # negative logits is contrast query and queue
        # queue is updated by key
        # ==================================================
        l_pos = self.compute_data_prob(q, k)  # (bs, 1)
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.t())  # (K, dim) * (dim, bs)
        l_neg = l_neg.t()  # (bs, K)

        out = torch.cat((l_pos, l_neg), dim=1)  # (bs, K+1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(out / self.T)
            if Z < 0:
                # initialize it with mean of first batch
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print('normalization constant Z is set to {:.1f}'.format(Z))
            else:
                Z_new = out.mean() * self.outputSize
                self.params[0] = (1 - self.params[1]) * Z_new + self.params[1] * self.params[0]
                Z = self.params[0].clone().detach().item()

            out = torch.div(out, Z).contiguous()
            probs = self.extract_probs(out)

        # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            # use mean of clips_num clips to replace each sample
            k_mean = torch.mean(k[self.indices], dim=1)  # (batchSize, clips_num, dim) -> (batchSize, dim)
            self.memory.index_copy_(0, out_ids, k_mean)
            self.index = (self.index + batchSize) % self.queueSize

        return out, probs
