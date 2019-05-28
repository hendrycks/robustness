# -*- coding: utf-8 -*-

import argparse
import os
import time
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
from resnext_50_32x4d import resnext_50_32x4d
from resnext_101_32x4d import resnext_101_32x4d
from resnext_101_64x4d import resnext_101_64x4d
from densenet_cosine_264_k48 import densenet_cosine_264_k48
from condensenet_converted import CondenseNet

parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Architecture
parser.add_argument('--model-name', '-m', type=str,
                    choices=['alexnet', 'squeezenet1.0', 'squeezenet1.1', 'condensenet4', 'condensenet8',
                             'vgg11', 'vgg', 'vggbn',
                             'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet264',
                             'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                             'resnext50', 'resnext101', 'resnext101_64'])
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
args = parser.parse_args()
print(args)

# /////////////// Model Setup ///////////////

if args.model_name == 'alexnet':
    net = models.AlexNet()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 256

elif args.model_name == 'squeezenet1.0':
    net = models.SqueezeNet(version=1.0)
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 256

elif args.model_name == 'squeezenet1.1':
    net = models.SqueezeNet(version=1.1)
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 256

elif args.model_name == 'condensenet4':
    args.evaluate = True
    args.stages = [4,6,8,10,8]
    args.growth = [8,16,32,64,128]
    args.data = 'imagenet'
    args.num_classes = 1000
    args.bottleneck = 4
    args.group_1x1 = 4
    args.group_3x3 = 4
    args.reduction = 0.5
    args.condense_factor = 4
    net = CondenseNet(args)
    state_dict = torch.load('./converted_condensenet_4.pth')['state_dict']
    for i in range(len(state_dict)):
        name, v = state_dict.popitem(False)
        state_dict[name[7:]] = v     # remove 'module.' in key beginning
    net.load_state_dict(state_dict)
    args.test_bs = 256

elif args.model_name == 'condensenet8':
    args.evaluate = True
    args.stages = [4,6,8,10,8]
    args.growth = [8,16,32,64,128]
    args.data = 'imagenet'
    args.num_classes = 1000
    args.bottleneck = 4
    args.group_1x1 = 8
    args.group_3x3 = 8
    args.reduction = 0.5
    args.condense_factor = 8
    net = CondenseNet(args)
    state_dict = torch.load('./converted_condensenet_8.pth')['state_dict']
    for i in range(len(state_dict)):
        name, v = state_dict.popitem(False)
        state_dict[name[7:]] = v     # remove 'module.' in key beginning
    net.load_state_dict(state_dict)
    args.test_bs = 256

elif 'vgg' in args.model_name:
    if 'bn' not in args.model_name:
        net = models.vgg19()
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
                                               model_dir='/share/data/lang/users/dan/.torch/models'))
    elif '11' in args.model_name:
        net = models.vgg11()
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
                                               model_dir='/share/data/lang/users/dan/.torch/models'))
    else:
        net = models.vgg19_bn()
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
                                               model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 64

elif args.model_name == 'densenet121':
    net = models.densenet121()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet121-a639ec97.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 128

elif args.model_name == 'densenet169':
    net = models.densenet169()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 128

elif args.model_name == 'densenet201':
    net = models.densenet201()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet201-c1103571.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 64

elif args.model_name == 'densenet161':
    net = models.densenet161()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet161-8d451a50.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 64

elif args.model_name == 'densenet264':
    net = densenet_cosine_264_k48
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet_cosine_264_k48.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 64

elif args.model_name == 'resnet18':
    net = models.resnet18()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 256

elif args.model_name == 'resnet34':
    net = models.resnet34()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 128

elif args.model_name == 'resnet50':
    net = models.resnet50()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 128

elif args.model_name == 'resnet101':
    net = models.resnet101()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 128

elif args.model_name == 'resnet152':
    net = models.resnet152()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet152-b121ed2d.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 64

elif args.model_name == 'resnext50':
    net = resnext_50_32x4d
    net.load_state_dict(torch.load('/share/data/lang/users/dan/.torch/models/resnext_50_32x4d.pth'))
    args.test_bs = 64

elif args.model_name == 'resnext101':
    net = resnext_101_32x4d
    net.load_state_dict(torch.load('/share/data/lang/users/dan/.torch/models/resnext_101_32x4d.pth'))
    args.test_bs = 64

elif args.model_name == 'resnext101_64':
    net = resnext_101_64x4d
    net.load_state_dict(torch.load('/share/data/lang/users/dan/.torch/models/resnext_101_64x4d.pth'))
    args.test_bs = 64

args.prefetch = 4

for p in net.parameters():
    p.volatile = True

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

torch.manual_seed(1)
np.random.seed(1)
if args.ngpu > 0:
    torch.cuda.manual_seed(1)

net.eval()
cudnn.benchmark = True  # fire on all cylinders

print('Model Loaded')

# /////////////// Data Loader ///////////////

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

clean_loader = torch.utils.data.DataLoader(dset.ImageFolder(
    root="/share/data/vision-greg/ImageNet/clsloc/images/val",
    transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])),
    batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)


# /////////////// Further Setup ///////////////

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


# correct = 0
# for batch_idx, (data, target) in enumerate(clean_loader):
#     data = V(data.cuda(), volatile=True)
#
#     output = net(data)
#
#     pred = output.data.max(1)[1]
#     correct += pred.eq(target.cuda()).sum()
#
# clean_error = 1 - correct / len(clean_loader.dataset)
# print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))


def show_performance(distortion_name):
    errs = []

    for severity in range(1, 6):
        distorted_dataset = dset.ImageFolder(
            root='/share/data/vision-greg/DistortedImageNet/JPEG/' + distortion_name + '/' + str(severity),
            transform=trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]))

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        correct = 0
        for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
            data = V(data.cuda(), volatile=True)

            output = net(data)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.cuda()).sum()

        errs.append(1 - 1.*correct / len(distorted_dataset))

    print('\n=Average', tuple(errs))
    return np.mean(errs)


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
import collections

print('\nUsing ImageNet data')

distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]

error_rates = []
for distortion_name in distortions:
    rate = show_performance(distortion_name)
    error_rates.append(rate)
    print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))


print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)))

