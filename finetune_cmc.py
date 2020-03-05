import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import time
import torch.backends.cudnn as cudnn
import argparse
import os
import json
import pdb
import itertools
import torchvision.transforms as transforms

from utils import AverageMeter, Logger, calculate_accuracy, ContrastiveSampler
from mean import get_mean, get_std
from model import generate_model
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel
from dataset import get_training_set, get_validation_set
from NCE import NCEAverage, NCECriterion
from test import kNN
import augmentation


def parse_args():
    parser = argparse.ArgumentParser(description='cmc finetuning on Videos')
    parser.add_argument('--video_path', default='', type=str)
    parser.add_argument('--annotation_path', default='', type=str)
    parser.add_argument('--result_path', default='./results', type=str)
    parser.add_argument('--modality', default='RGB', type=str, help='Modality of input data. RGB, Flow or RGBFlow')
    parser.add_argument('--dataset', default='kinetics', type=str, help='Used dataset (activitynet | kinetics | ucf101 | hmbd51')
    parser.add_argument('--n_classes', default=400, type=int, help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str, help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--learning_rate', default=0.04, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--mean_dataset', default='activitynet', type=str, help='dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=250, type=int, help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default=None, type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument('--n_threads', default=16, type=int, help='num of workers loading dataset')
    parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--model', default='resnet', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--checkpoint', default=10, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='iter to enable large batch size training')
    parser.add_argument('--lr_patience', default=3, type=int, help='Patience of LR scheduler')
    parser.add_argument('--nce_k', default=4096, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--low_dim', default=128, type=int, help='dimension of embeddings')
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed every epoch')
    parser.set_defaults(no_val=False)
    parser.add_argument('--no_tracking', action='store_true', help='If true, BN uses tracking running stats')
    parser.set_defaults(no_tracking=False)
    parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--test_size', default=16, type=int, help='batch size for validation loader')
    parser.add_argument('--trained_model', default=None, type=str, help='model pth for running test')
    parser.add_argument('--class_num', default=80, type=int, help='number of different classes within a batch')
    parser.add_argument('--sample_num', default=2, type=int, help='number of samples for each class within a batch')
    parser.add_argument('--ft_portion', default='fc', type=str,
                        help='specify finetune whole model or simple the top layer, fc or complete')
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--test_freq', default=5, type=int)

    args = parser.parse_args()

    return args


class LinearModel(nn.Module):
    def __init__(self, args):
        super(LinearModel, self).__init__()
        self.num_infeatures = args.low_dim
        # self.classifier = nn.Linear(self.num_infeatures, args.n_classes)
        self.classifier = nn.Sequential(nn.BatchNorm1d(self.num_infeatures, affine=False),
                                        nn.Linear(self.num_infeatures, args.n_classes))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)


class AssembleModel(nn.Module):
    def __init__(self, args, backbone):
        super(AssembleModel, self).__init__()
        self.backbone = backbone
        if args.pretrain_path is not None:
            pretrain = torch.load(args.pretrain_path)
            self.backbone.load_state_dict(pretrain['state_dict'])
            print('==> loaded checkpoint {} (epoch {})'.format(args.pretrain_path, pretrain['epoch']))
        else:
            raise ValueError('Please specify a pretrain path for finetuning!')

        self.num_infeatures = args.low_dim
        self.classifier = nn.Sequential(nn.BatchNorm1d(self.num_infeatures, affine=False),
                                        nn.Linear(self.num_infeatures, args.n_classes)).cuda()

    def forward(self, x):
        avg_f, _, _ = self.backbone(x)
        out = self.classifier(avg_f)

        return out


def set_lr(optimizer, lr_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_rate


def neq_load_customized(model, pretrained_dict):
    model_dict = model.state_dict()
    tmp = {}
    print('\n=======Check Weights Loading======')
    print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            print(k)
    print('---------------------------')
    print('Weights not loaded into new model:')
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def set_model(args):
    model, _ = generate_model(args)
    classifier = LinearModel(args).cuda()

    return model, classifier


def train(epoch, train_loader, model, classifier, criterion, optimizer, args,
          epoch_logger, batch_logger):
    # model.eval()
    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # ct_losses = AverageMeter()
    # ce_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # probs = AverageMeter()

    end = time.time()
    # average.prepare_indices(args.bat)
    for idx, (input, target, idxs) in enumerate(train_loader):
        # pdb.set_trace()
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        idxs = idxs.cuda()
        bs = input.size(0)

        # with torch.no_grad():
        #     features = model(input)
        # with torch.no_grad():
        #     avg_f, _, _ = model(input)
        #     avg_f = avg_f.detach()
        avg_f, _, f = model(input)

        output = classifier(avg_f)
        # output = model(input)
        # loss = criterion(output, target)
        # out, prob = average(f, idxs)
        # uprob = out[:, 0].mean().detach().item()
        # loss = criterion['contrastive'](out)
        loss = criterion['cross_entropy'](output, target)
        # loss = contrastive_loss + ce_loss

        acc1, acc5 = calculate_accuracy(output.detach(), target.detach(), topk=(1, 5))

        # ct_losses.update(ct_loss.item(), bs)
        # ce_losses.update(ce_loss.item(), bs)
        losses.update(loss.item(), bs)
        top1.update(acc1.item(), bs)
        top5.update(acc5.item(), bs)
        # probs.update(prob.item(), bs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': idx + 1,
            'iter': (epoch-1) * len(train_loader) + i+1,
            # 'contrastive': ct_losses.val,
            'crossentropy': losses.val,
            # 'prob': probs.val,
            # 'contrastive': losses.val,
            'prec1': top1.val,
            'prec5': top5.val,
            # 'backbone_lr': optimizer.param_groups[0]['lr'],
            # 'fc_lr': optimizer.param_groups[1]['lr'],
            'lr': optimizer.param_groups[0]['lr']
        })
        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                  'Crossentropy {Crossentropy.val:.4f} ({Crossentropy.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch,
                      idx,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      Crossentropy=losses,
                      top1=top1,
                      top5=top5))

    epoch_logger.log({
        'epoch': epoch,
        # 'contrastive': losses.avg,
        'crossentropy': losses.avg,
        # 'prob': probs.avg,
        # 'loss': losses.avg,
        'prec1': top1.avg,
        'prec5': top5.avg,
        # 'backbone_lr': optimizer.param_groups[0]['lr'],
        # 'fc_lr': optimizer.param_groups[1]['lr'],
        # 'fc_lr': optimizer.param_groups[0]['lr']
        'lr': optimizer.param_groups[0]['lr']
    })

    return top1.avg, top5.avg, losses.avg


def validate(epoch, val_loader, model, classifier, criterion, args, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            avg_f, _, _ = model(input)
            output = classifier(avg_f)
            # output = model(input)
            loss = criterion['cross_entropy'](output, target)

            acc1, acc5 = calculate_accuracy(output.detach(), target.detach(), topk=(1, 5))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            losses.update(loss.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                      'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                      'Crossentropy {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                      'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                    epoch,
                    idx + 1,
                    len(val_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5))

    logger.log({
        'epoch': epoch,
        'crossentropy': losses.avg,
        'prec1': top1.avg,
        'prec5': top5.avg,
        # 'backbone_lr': optimizer.param_groups[0]['lr'],
        # 'fc_lr': optimizer.param_groups[1]['lr'],
        # 'fc_lr': optimizer.param_groups[0]['lr']
    })

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    args.scales = [args.initial_scale]
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scale_step)
    args.mean = get_mean(args.norm_value, dataset=args.mean_dataset)
    args.std = get_std(args.norm_value)

    print(args)
    with open(os.path.join(args.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)

    torch.manual_seed(args.manual_seed)

    print('\n==> Building Model...')
    model, classifier = set_model(args)
    # model, _ = generate_model(args)
#     backbone, _ = generate_model(args)
    # model = AssembleModel(args, backbone)

    # print('\n==> Freeze backbone...')
    # for param in model.parameters():
    #     param.requires_grad = False

    # criterion = nn.CrossEntropyLoss().cuda()

    if args.no_mean_norm and not args.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not args.std_norm:
        norm_method = Normalize(args.mean, [1, 1, 1])
    else:
        norm_method = Normalize(args.mean, args.std)

    assert args.train_crop in ['random', 'corner', 'center']
    if args.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(args.scales, args.sample_size)
    elif args.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(args.scales, args.sample_size)
    elif args.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            args.scales, args.sample_size, crop_positions=['c'])

    print('==> Preparing dataset...')
    print('\nTraining set')
    spatial_transform = Compose([
        Gaussian_blur(p=0.5),
        RandomHorizontalFlip(),
        crop_method,
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3),
        ToTensor(args.norm_value), norm_method
    ])
    # ================ transform as DPC =====================
    # spatial_transform = transforms.Compose([
    #     augmentation.RandomHorizontalFlip(consistent=True),
    #     augmentation.MultiScaleRandomCrop(args.scales, args.sample_size, consistent=True),
    #     augmentation.RandomHorizontalFlip(consistent=True),
    #     augmentation.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
    #     augmentation.ToTensor(),
    #     augmentation.Normalize()
    # ])
    temporal_transform = TemporalRandomCrop(args.sample_duration, args.downsample)
    target_transform = ClassLabel()
    training_data = get_training_set(args, spatial_transform, temporal_transform, target_transform)
    # batchsampler = ContrastiveSampler(training_data.lookup, args.class_num, args.sample_num, len(training_data))
    # train_loader = torch.utils.data.DataLoader(
    #     training_data,
    #     batch_sampler=batchsampler,
    #     num_workers=args.n_threads,
    #     pin_memory=True,
    # )
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=True
    )
    if not args.no_val:
        print('\nValidation set')
        spatial_transform = Compose([
            Scale(args.sample_size),
            CenterCrop(args.sample_size),
            # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            ToTensor(args.norm_value),
            norm_method
        ])
        # ================ transform as DPC ======================
        # spatial_transform = transforms.Compose([
        #     augmentation.Scale(size=args.sample_size),
        #     augmentation.CenterCrop(size=args.sample_size, consistent=True),
        #     augmentation.ToTensor(),
        #     augmentation.Normalize()
        # ])
        temporal_transform = TemporalCenterCrop(args.sample_duration, args.downsample)
        target_transform = ClassLabel()
        validation_data = get_validation_set(args, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=args.test_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True
        )

    # ndata = len(training_data)
    # average = NCEAverage.Average(args.low_dim, ndata, args.sample_num,
    #                              args.class_num, args.nce_t).cuda()
    # criterion_ct = NCECriterion.NCECriterion(ndata).cuda()
    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion = {}
    # criterion['contrastive'] = criterion_ct
    criterion['cross_entropy'] = criterion_ce
    # optimizer = torch.optim.Adam(
    #     [{'params': model.parameters(), 'lr': args.learning_rate / 10, 'weight_decay': args.weight_decay},
    #      {'params': classifier.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-3}])
    # if args.nesterov:
    #     dampening = 0
    # else:
    #     dampening = args.dampening
    optimizer = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    cudnn.benchmark = True
    if args.resume_path is not None:
        resume_ckpt = torch.load(args.resume_path)
        model.load_state_dict(resume_ckpt['state_dict'])
        classifier.load_state_dict(resume_ckpt['classifier'])
        args.begin_epoch = resume_ckpt['epoch'] + 1
        best_acc = resume_ckpt['best_acc']
        optimizer.load_state_dict(resume_ckpt['optimizer'])

        print('==> Resume training...')
        print('best acc is: {}'.format(best_acc))
        del resume_ckpt
        torch.cuda.empty_cache()
        # adjust_learning_rate(optimizer, 1, 0.0001)
        set_lr(optimizer, 0.001)
        resume = True
    else:
        print('==> Train from sratch...')
        resume = False
        best_acc = 0
        print('==> loading pre-trained model and NCE')
        ckpt = torch.load(args.pretrain_path)
        try:
            model.load_state_dict(ckpt['state_dict'])
        except:
            print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
            model = neq_load_customized(model, ckpt['state_dict'])
        # average.load_state_dict(ckpt['average'])
        print('==> loaded checkpoint {} (epoch {})'.format(args.pretrain_path, ckpt['epoch']))
        # print('==> [NCE]: params Z {}'.format(average.params[0].item()))
        print('==> done')
        del ckpt
        torch.cuda.empty_cache()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           patience=args.lr_patience,
                                                           verbose=True,
                                                           threshold=0.1,
                                                           threshold_mode='abs')
    train_batch_logger = Logger(os.path.join(args.result_path, 'train_batch.log'),
                          ['epoch', 'batch', 'iter', 'crossentropy', 'prec1', 'prec5', 'lr'], resume)
    train_logger = Logger(os.path.join(args.result_path, 'train.log'),
                                ['epoch', 'crossentropy', 'prec1', 'prec5', 'lr'], resume)

    if not args.no_val:
        val_logger = Logger(os.path.join(args.result_path, 'val.log'),
                            ['epoch', 'crossentropy', 'prec1', 'prec5'], resume)

    print('\nrun')
    for epoch in range(args.begin_epoch, args.n_epochs):
        print('\nEpoch {}'.format(epoch))
        # if epoch == 50:
        #     print('==> Decay lr by 0.1 after 50 epochs...')
        #     adjust_learning_rate(optimizer, 0.01, 0.001)
        train_acc1, train_acc5, train_loss = train(epoch, train_loader, model, classifier, criterion,
                                                   optimizer, args, train_logger, train_batch_logger)
        # scheduler.step(train_loss)
        if not args.no_val and epoch % args.test_freq == 0:
            print('\nValidation...')
            val_acc1, val_acc5, val_loss = validate(epoch, val_loader, model, classifier, criterion, args, val_logger)
            scheduler.step(val_acc1)
            val_logger.log({
                'epoch': epoch,
                'crossentropy': val_loss,
                'prec1': val_acc1,
                'prec5': val_acc5
            })
        #     val_acc1, val_acc5 = kNN(args, args.n_classes, model, average, train_loader, val_loader, 200, 0)
        # scheduler.step(val_acc1)
            if val_acc1 > best_acc:
                best_acc = val_acc1
                print('==> Saving...')
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'classifier': classifier.state_dict(),
                }
                save_name = os.path.join(args.result_path, 'best.pth')
                torch.save(state, save_name)
