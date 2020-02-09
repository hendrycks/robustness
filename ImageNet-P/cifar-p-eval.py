# An example cifar-10/100-p evaluation script; some imports may not work out-of-the-box
# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.wrn import WideResNet
from models.resnext import resnext29
from models.densenet import densenet
from models.allconv import AllConvNet


parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', '-d', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='resnext',
                    choices=['wrn', 'allconv', 'densenet', 'resnext'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/adv', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots/augmix', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

# # mean and standard deviation of channels of CIFAR-10 images
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor()])
test_transform = trn.Compose([trn.ToTensor()])

if args.dataset == 'cifar10':
    train_data = dset.CIFAR10('~/datasets/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR10('~/datasets/cifarpy', train=False, transform=test_transform)
    num_classes = 10
else:
    train_data = dset.CIFAR100('~/datasets/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR100('~/datasets/cifarpy', train=False, transform=test_transform)
    num_classes = 100


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model == 'densenet':
    args.decay = 0.0001
    args.epochs = 200
    net = densenet(num_classes=num_classes)
elif args.model == 'wrn':
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
elif args.model == 'allconv':
    net = AllConvNet(num_classes)
elif args.model == 'resnext':
    args.epochs = 200
    net = resnext29(num_classes=num_classes)

state = {k: v for k, v in args._get_kwargs()}
print(state)

start_epoch = 0

# Restore model if desired
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset + '_' + args.model +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
        model_name = os.path.join(args.load, args.dataset + '_' + args.model + '_' + args.model +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

net.eval()

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.to('cpu').numpy()

def evaluate(loader):
    confidence = []
    correct = []

    num_correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.cuda(), target.cuda()

            output = net(2 * data - 1)

            # accuracy
            pred = output.data.max(1)[1]
            num_correct += pred.eq(target.data).sum().item()

            confidence.extend(to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist())
            pred = output.data.max(1)[1]
            correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())

    return num_correct / len(loader.dataset), np.array(confidence), np.array(correct)


acc, test_confidence, test_correct = evaluate(test_loader)
print('Error', 100 - 100. * acc)
print('RMS', 100 * calib_err(test_confidence, test_correct, p='2'))
# print('AURRA', 100 * aurra(test_confidence, test_correct))

# /////////////// Stability Measurements ///////////////

args.difficulty = 1
identity = np.asarray(range(1, num_classes+1))
cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (num_classes-1 - 5)))
recip = 1./identity


def dist(sigma, mode='top5'):
    if mode == 'top5':
        return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma-1][:5]))
    elif mode == 'zipf':
        return np.sum(np.abs(recip - recip[sigma-1])*recip)


def ranking_dist(ranks, noise_perturbation=False, mode='top5'):
    result = 0
    step_size = 1 if noise_perturbation else args.difficulty

    for vid_ranks in ranks:
        result_for_vid = []

        for i in range(step_size):
            perm1 = vid_ranks[i]
            perm1_inv = np.argsort(perm1)

            for rank in vid_ranks[i::step_size][1:]:
                perm2 = rank
                result_for_vid.append(dist(perm2[perm1_inv], mode))
                if not noise_perturbation:
                    perm1 = perm2
                    perm1_inv = np.argsort(perm1)

        result += np.mean(result_for_vid) / len(ranks)

    return result


def flip_prob(predictions, noise_perturbation=False):
    result = 0
    step_size = 1 if noise_perturbation else args.difficulty

    for vid_preds in predictions:
        result_for_vid = []

        for i in range(step_size):
            prev_pred = vid_preds[i]

            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation: prev_pred = pred

        result += np.mean(result_for_vid) / len(predictions)

    return result


# /////////////// Get Results ///////////////

from tqdm import tqdm
from scipy.stats import rankdata

c_p_dir =  'CIFAR-10-P' if num_classes == 10 else 'CIFAR-100-P'
c_p_dir = '/home/hendrycks/datasets/' + c_p_dir


dummy_targets = torch.LongTensor(np.random.randint(0, num_classes, (10000,)))

flip_list = []
zipf_list = []

for p in ['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
          'spatter', 'brightness', 'translate', 'rotate', 'tilt', 'scale']:
    # ,'speckle_noise', 'gaussian_blur', 'snow', 'shear']:
    dataset = torch.from_numpy(np.float32(np.load(os.path.join(c_p_dir, p + '.npy')).transpose((0,1,4,2,3))))/255.

    ood_data = torch.utils.data.TensorDataset(dataset, dummy_targets)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=25, shuffle=False, num_workers=2, pin_memory=True)

    predictions, ranks = [], []

    with torch.no_grad():

        for data in loader:
            num_vids = data.size(0)
            data = data.view(-1,3,32,32).cuda()

            output = net(data * 2 - 1)

            for vid in output.view(num_vids, -1, num_classes):
                predictions.append(vid.argmax(1).to('cpu').numpy())
                ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])

        ranks = np.asarray(ranks)

        # print('\nComputing Metrics for', p,)

        current_flip = flip_prob(predictions, True if 'noise' in p else False)
        current_zipf = ranking_dist(ranks, True if 'noise' in p else False, mode='zipf')
        flip_list.append(current_flip)
        zipf_list.append(current_zipf)

        print('\n' + p, 'Flipping Prob')
        print(current_flip)
        # print('Top5 Distance\t{:.5f}'.format(ranking_dist(ranks, True if 'noise' in p else False, mode='top5')))
        # print('Zipf Distance\t{:.5f}'.format(current_zipf))

print(flip_list)
print('\nMean Flipping Prob\t{:.5f}'.format(np.mean(flip_list)))
# print('Mean Zipf Distance\t{:.5f}'.format(np.mean(zipf_list)))
