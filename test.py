import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json
import pdb

from utils import AverageMeter, pearson_coefficient_bank, show_distribution


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[int(locs[i])],
            'score': float(sorted_scores[i])
        })

    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            inputs = Variable(inputs)
        outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs, dim=1)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]

        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}.json'.format(
                        opt.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)


def kNN(args, C, model, average, trainloader, testloader, K, recompute_memory=0):
    model.eval()
    model_time = AverageMeter()
    cluster_time = AverageMeter()
    total = 0

    testsize = testloader.dataset.__len__()
    ndata = trainloader.dataset.__len__()

    if recompute_memory:
        trainFeatures = torch.zeros(ndata, args.low_dim).cuda()
    else:
        trainFeatures = average.memory  # (num_samples, low_dim)

    # this is cifar10
    # trainLabels = torch.tensor(trainloader.dataset.targets).long().cuda()

    # this is for UCF101 or Kinetics
    trainLabels = torch.tensor([sample['label'] for sample in trainloader.dataset.data]).long().cuda()

    if recompute_memory:
        print('\nRecomputing memory bank....')
        # use test transform to go through all train samples and retrieve features as memory
        # transform_bak = trainloader.dataset.transform
        # trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.n_threads,
                                                 pin_memory=True)
        memory_idx = torch.arange(args.batch_size * args.clips_num).view(args.clips_num, args.batch_size).t().cuda()
        batchSize = args.batch_size
        with torch.no_grad():
            for batch_idx, (inputs, _, indices) in enumerate(temploader):
                inputs = torch.cat(inputs, dim=0)
                inputs = inputs.cuda()

                bs = inputs.size(0)

                _, _, features = model(inputs)

                if batch_idx == len(temploader) - 1:
                    batch_size = bs // args.clips_num
                    memory_idx = torch.arange(bs).view(args.clips_num, batch_size).t().cuda()
                    f_means = torch.mean(features[memory_idx], dim=1)
                    trainFeatures[batch_idx * batchSize:, :] = f_means
                else:
                    f_means = torch.mean(features[memory_idx], dim=1)  # (batchSize, dim)
                    trainFeatures[batch_idx * batchSize : (batch_idx+1) * batchSize, :] = f_means
        # trainloader.dataset.transform = transform_bak
        print('Finished!')

    top1 = 0
    top5 = 0
    # save plt distribution
    # Yd = torch.zeros(testsize, K).cuda()
    # NN_labels = torch.zeros(testsize, K).long().cuda()
    # labels = torch.zeros(testsize).long().cuda()

    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, _) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda()
            inputs = inputs.cuda()

            batchSize = inputs.size(0)
            _, _, features = model(inputs)
            total += targets.size(0)

            model_time.update(time.time() - end)
            end = time.time()

            # dist = pearson_coefficient_bank(features, trainFeatures.t())
            dist = torch.mm(features, trainFeatures.t())
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            # if batch_idx == 0:
            #     show_distribution(yd.cpu(), retrieval.cpu(), targets.cpu())
            # if batch_idx < len(testloader)-1:
            #     Yd[batch_idx*batchSize : (batch_idx+1)*batchSize, :] = yd
            #     NN_labels[batch_idx*batchSize : (batch_idx+1)*batchSize, :] = retrieval
            #     labels[batch_idx*batchSize : (batch_idx+1)*batchSize] = targets
            # else:
            #     Yd[batch_idx*batchSize:, :] = yd
            #     NN_labels[batch_idx*batchSize:, :] = retrieval
            #     labels[batch_idx*batchSize:] = targets

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1) # inverse operation of torch.gather
            yd_transform = torch.exp(torch.div(yd, args.nce_t)) # apply softmax with temperature for (non-parametric logits)

            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # find which predictions match the target
            correct = predictions.eq(targets.view(-1, 1))
            cluster_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()

            if (batch_idx+1) % 100 == 0:
                print('Test [{}/{}]\t'
                      'Model time: {model_time.val:.3f} ({model_time.avg:.3f})\t'
                      'Cluster time: {cluster_time.val:.3f} ({cluster_time.avg:.3f})\t'
                      'Top1: {:.2f} Top5: {:.2f}'.format(batch_idx+1, len(testloader), top1*100./total, top5*100./total,
                                                         model_time=model_time, cluster_time=cluster_time))

    print(top1*100./total)

    return top1*100./total, top5*100./total


def NN(args, model, lemniscate, trainloader, testloader, recompute_memory=0):
    model.eval()
    model_time = AverageMeter()
    cluster_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t() # (low_dim, num_samples)
    trainLabels = torch.tensor([sample['label'] for sample in trainloader.dataset.data]).long().cuda()
    # if hasattr(trainloader.dataset, 'imgs'):
    #     trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    # else:
    #     trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset,
                                                 batch_size=args.test_size,
                                                 shuffle=False,
                                                 num_workers=args.n_threads,
                                                 pin_memory=True)
        for batch_idx, (inputs, _, indexes) in enumerate(temploader):
            inputs = inputs.cuda()
            batchSize = inputs.size(0)
            features = model(inputs)
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.detach().t()
        # trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak

    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()

            inputs = inputs.cuda()
            targets = targets.cuda()
            batchSize = inputs.size(0)
            features = model(inputs)
            model_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()

            cluster_time.update(time.time() - end)

            print('Test [{}/{}]\t'
                  'Net Time {model_time.val:.3f} ({model_time.avg:.3f})\t'
                  'Cls Time {cluster_time.val:.3f} ({cluster_time.avg:.3f})\t'
                  'Top1: {:.2f}'.format(total, testsize, correct * 100. / total,
                                        model_time=model_time, cluster_time=cluster_time))

    return correct / total
