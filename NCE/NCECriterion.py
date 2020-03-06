import torch
import torch.nn as nn
from NCE.NCEAverage import NCEAverage

from utils import l2_normalize

eps = 1e-7


class NCECriterion(nn.Module):
    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.size(0)
        K = x.size(1) - 1

        # noise distribution
        Pn = 1. / self.n_data

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(K * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, K)
        log_D0 = torch.div(P_neg.clone().fill_(K * Pn), P_neg.add(K * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class KLCriterion(nn.Module):
    def __init__(self, clips_num):
        super(KLCriterion, self).__init__()
        self.clips_num = clips_num

    def forward(self, x, pos_indices):
        bsz = x.size(0)
        assert bsz == pos_indices.size(0), 'FATAL ERROR!'
        # batchSize = bsz // self.clips_num

        x_other = x[pos_indices]  # (bsz, clips_num-1, x.size(1))
        x_other = torch.mean(x_other, dim=1)  # same shape as x

        # input_log_softmax = torch.nn.functional.log_softmax(x, dim=1)
        input_log = torch.log(x)
        # target_softmax = torch.nn.functional.softmax(x_other, dim=1)

        return torch.nn.functional.kl_div(input_log, x_other, reduction='batchmean')



if __name__ == '__main__':
    average = NCEAverage(128, 9000, 1000, 4, 0.07, 3).cuda()
    criterion = NCECriterion(9000).cuda()
    kl_criterion = KLCriterion(4).cuda()
    # print(average.pos_indices)
    # print(average.neg_indices)
    dummy_embeddings = torch.randn(12, 128).cuda()
    dummy_embeddings = l2_normalize(dummy_embeddings)
    idxs = torch.arange(3).cuda()

    outs, probs= average(dummy_embeddings, idxs, 0)
    print(outs[:, 0].mean())
    print(probs)
    loss = criterion(outs).item()
    print(loss)
    kl_loss = kl_criterion(outs, average.pos_indices).item()
    print(kl_loss)
    # print(kl_loss.shape)
