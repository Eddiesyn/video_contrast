import math
import torch
import torch.nn as nn
import numpy as np
import pdb

from NCE.alias_multinomial import AliasMethod
from utils import l2_normalize, show_distribution


# ==========================
# Instance Discrimination
# ==========================

class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, trainLabel, T=0.07, momentum=0.5, z_momentum=0.9, use_softmax=False):
        '''
        inputSize: feature dimensions
        outputSize: number of samples in dataset
        K: number of noise samples
        T: temperature of softmax
        momentum: update rate of memory bank
        Z_momentum: update rate of normalization factor
        '''
        super(MemoryInsDis, self).__init__()
        print('[InsDis]: Initialization...')
        self.outputSize = outputSize
        self.inputSize = inputSize

        # Alias Method to draw negative samples
        self.unigrams = torch.ones(self.outputSize)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()

        self.trainLabel = trainLabel
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, momentum, z_momentum]))
        print('[InsDis]: params K {}, T {}, Z {}, momentum {}, Z_momentum {}'.format(K, T, -1, momentum, z_momentum))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('[InsDis]: memory bank shape: {}, {}'.format(outputSize, inputSize))

    def extract_probs(self, out):
        probs = out / torch.sum(out, dim=1, keepdim=True)

        return probs[:, 0].mean()

    # def nce_core_ms(self, out, K):
    #     '''another loss based on output of nce_core'''
    #     yd, yi = out.topk(32, dim=1, largest=True, sorted=True)  # select top 32 similar
    #     weights = self.get_weights(yd)
    #     weights = torch.nn.functional.normalize(weights, p=1, dim=1)
    #     # weighted sum of neighbour features
    #

    def nce_core(self, x, y, idx, K):
        '''x is network's output feature'''
        bs = x.size(0)
        if idx is None:
            idx = self.multinomial.draw(bs * (K+1)).view(bs, -1)  # each sample draw K noise samples
            idx.select(1, 0).copy_(y.data)
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(bs, K+1, self.inputSize)
        # (bs, K+1, dim) @ (bs, dim, 1) = (bs, K+1, 1)
        out = torch.bmm(weight, x.view(bs, self.inputSize, 1))

        return out

    def get_weights(self, out):
        weights = (1 + torch.exp(-60 * (out-0.5))).pow(-1)

        return weights

    def forward(self, x, target, y, idx=None):
        '''
        x: featrues
        y: indices
        '''
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()
        z_momentum = self.params[4].item()

        # pdb.set_trace()

        bs = x.size(0)

        # nce_core
        nce_core_out = self.nce_core(x, y, idx, K)

        # NN statistics
        out = torch.mm(x, self.memory.t())  # (bs, dim) (dim, num)
        yd, yi = out.topk(32, dim=1, largest=True, sorted=True)
        candidates = self.trainLabel.view(1, -1).expand(bs, -1)
        retrieval = torch.gather(candidates, 1, yi)

        show_distribution(yd.cpu(), retrieval.cpu(), target.cpu())
        pdb.set_trace()

        out = self.nce_core_ms(out, K)

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
            probs = self.extract_probs(out)
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                # initialize it with mean of first batch
                self.params[2] = out.mean() * self.outputSize
                Z = self.params[2].item()
                print('normalization constant Z is set to {:.1f}'.format(Z))
            # else:
            #     # update normalization factor with z_momentum
            #     Z_new = out.mean() * self.outputSize
            #     self.params[2] = z_momentum * Z_new + (1 - z_momentum) * self.params[2]
            #     Z = self.params[2].clone().detach().item()
            out = torch.div(out, Z).squeeze().contiguous()
            probs = self.extract_probs(out)

        # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1-momentum))
            # l2 normalize it again
            normed_weight = l2_normalize(weight_pos)
            self.memory.index_copy_(0, y, normed_weight)

        return out, probs
