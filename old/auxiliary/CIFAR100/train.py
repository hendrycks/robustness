# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pickle
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.autograd import Variable as V
from cifar_resnet import WideResNet

parser = argparse.ArgumentParser(description='Trains a CIFAR-100 Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability (default: 0.0)')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()
args.dataset = 'cifar100'

torch.manual_seed(1)
np.random.seed(1)

state = {k: v for k, v in args._get_kwargs()}
state['tt'] = 0     # SGDR variable
state['init_learning_rate'] = args.learning_rate


# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])


train_data = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=True, transform=train_transform, download=False)
test_data = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=False, transform=test_transform, download=False)
test_data_out = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=False, transform=test_transform, download=False)
num_classes = 20

# d = dict.fromkeys([i for i in range(20)])
# for i in range(len(coarse)):
#     if d[coarse[i]] is None: d[coarse[i]] = []
#     if fine[i] not in d[coarse[i]]:
#         d[coarse[i]].append(fine[i])

coarse_to_fine =\
    {0: [72, 4, 95, 30, 55], 1: [73, 32, 67, 91, 1], 2: [92, 70, 82, 54, 62],
     3: [16, 61, 9, 10, 28], 4: [51, 0, 53, 57, 83], 5: [40, 39, 22, 87, 86],
     6: [20, 25, 94, 84, 5], 7: [14, 24, 6, 7, 18], 8: [43, 97, 42, 3, 88],
     9: [37, 17, 76, 12, 68], 10: [49, 33, 71, 23, 60], 11: [15, 21, 19, 31, 38],
     12: [75, 63, 66, 64, 34], 13: [77, 26, 45, 99, 79], 14: [11, 2, 35, 46, 98],
     15: [29, 93, 27, 78, 44], 16: [65, 50, 74, 36, 80], 17: [56, 52, 47, 59, 96],
     18: [8, 58, 90, 13, 48], 19: [81, 69, 41, 89, 85]}

# {v: k for k, v in coarse_to_fine.items()}
fine_to_coarse = dict((v,k) for k in coarse_to_fine for v in coarse_to_fine[k])

train_in_data = []
train_in_labels = []
for i in range(len(train_data)):
    fine = train_data.train_labels[i]

    if coarse_to_fine[fine_to_coarse[fine]].index(fine) > 0:    # 0, 1, 2, 3
        train_in_data.append(train_data.train_data[i])
        train_in_labels.append(fine_to_coarse[fine])

train_in_data = np.array(train_in_data)
train_data.train_data = train_in_data
train_data.train_labels = train_in_labels


test_in_data = []
test_in_labels = []
test_out_data = []
test_out_labels = []
for i in range(len(test_data)):
    fine = test_data.test_labels[i]

    if coarse_to_fine[fine_to_coarse[fine]].index(fine) > 0:
        test_in_data.append(test_data.test_data[i])
        test_in_labels.append(fine_to_coarse[fine])
    else:
        test_out_data.append(test_data.test_data[i])
        test_out_labels.append(fine_to_coarse[fine])

test_in_data = np.array(test_in_data)
test_data.test_data = test_in_data
test_data.test_labels = test_in_labels

test_out_data = np.array(test_out_data)
test_data_out.test_data = test_out_data
test_data_out.test_labels = test_out_labels


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)
test_out_loader = torch.utils.data.DataLoader(
    test_data_out, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

print('Number of in examples:',len(test_loader.dataset))
print('Number of out examples:', len(test_out_loader.dataset))

# Create model
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

start_epoch = 0

# # Restore model
# if args.load != '':
#     for i in range(1000 - 1, -1, -1):
#         model_name = os.path.join(args.load, args.dataset + '_model_epoch' + str(i) + '.pytorch')
#         if os.path.isfile(model_name):
#             net.load_state_dict(torch.load(model_name))
#             print('Model restored! Epoch:', i)
#             start_epoch = i + 1
#             break
#     if start_epoch == 0:
#         assert False, "could not resume"


if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                            weight_decay=state['decay'], nesterov=True)

from tqdm import tqdm

def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        t = torch.from_numpy(np.random.beta(1,1, size=data.size(0)).astype(np.float32)).view(-1,1,1,1)
        perm = torch.from_numpy(np.random.permutation(data.size(0))).long()

        data, target = V((t * data + (1 - t) * data[perm]).cuda()), V(target.cuda())

        # forward
        x = net(data)

        # backward
        optimizer.zero_grad()
        t = V(t.view(-1).cuda())
        loss = (t * F.cross_entropy(x, target, reduce=False)).mean() +\
               ((1 - t) * F.cross_entropy(x, target[V(perm.cuda())], reduce=False)).mean()
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + loss.data[0] * 0.2

        dt = math.pi / float(args.epochs)
        state['tt'] += float(dt) / (len(train_loader.dataset) / float(args.batch_size))
        if state['tt'] >= math.pi - 0.01:
            state['tt'] = math.pi - 0.01
        curT = math.pi / 2.0 + state['tt']
        new_lr = args.learning_rate * (1.0 + math.sin(curT)) / 2.0  # lr_min = 0, lr_max = lr
        state['learning_rate'] = new_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['learning_rate']

    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = V(data.cuda(), volatile=True), V(target.cuda(), volatile=True)

        # forward
        output = net(data)
        loss = F.cross_entropy(output, target)

        # accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum()

        # test loss average
        loss_avg += loss.data[0]

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


def test_out():
    net.eval()
    loss_avg = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_out_loader):
        data, target = V(data.cuda(), volatile=True), V(target.cuda(), volatile=True)

        # forward
        output = net(data)
        loss = F.cross_entropy(output, target)

        # accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum()

        # test loss average
        loss_avg += loss.data[0]

    state['test_out_loss'] = loss_avg / len(test_out_loader)
    state['test_out_accuracy'] = correct / len(test_out_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

state['learning_rate'] = state['init_learning_rate']

print('Beginning Training')
# Main loop
best_accuracy = 0.0
for epoch in range(start_epoch, args.epochs):
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['learning_rate']
    state['tt'] = math.pi / float(args.epochs) * epoch

    state['epoch'] = epoch

    begin_epoch = time.time()
    train()
    print('Epoch', epoch, '| Time Spent:', round(time.time() - begin_epoch, 4))

    test()
    test_out()

    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + '_model_subclass_epoch' + str(epoch) + '.pytorch'))
    # Let us not waste space and delete the previous model
    # We do not overwrite the model because we need the epoch number
    try: os.remove(os.path.join(args.save, args.dataset + '_model_subclass_epoch' + str(epoch - 1) + '.pytorch'))
    except: True

    print(state)
