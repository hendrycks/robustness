# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
import math
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.augment import RandomErasing

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test-style', type=str, default='microsoft')
parser.add_argument('--traditional', action='store_true', help='Test classification performance not robustness.')
parser.add_argument('--subtype', action='store_true', help='Test subtype robustness.')
parser.add_argument('--c100', action='store_true', help='Test classification performance on CIFAR-100 not robustness.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--valid_size', '-v', type=int, default=0, help='Number of validation examples to hold out.')
parser.add_argument('--test_bs', type=int, default=250)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0001, help='Weight decay (L2 penalty).')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Architecture
parser.add_argument('--model', default='resnext', type=str)
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
args = parser.parse_args()
args.dataset = 'icons'

state = {k: v for k, v in args._get_kwargs()}

print(state)

# set seeds
torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)


# /////////////// Dataset Loading ///////////////

if args.c100:
    # mean and standard deviation of channels of CIFAR-10 images
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                   trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    train_data = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=True, transform=train_transform, download=False)
    test_data = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=False, transform=test_transform, download=False)
    num_classes = 100

else:
    train_data = dset.ImageFolder('/share/data/vision-greg/DistortedImageNet/Icons-50',
                                  transform=trn.Compose([trn.Resize((32, 32)), trn.RandomHorizontalFlip(),
                                                         trn.RandomCrop(32, padding=4), trn.ToTensor(),
                                                         # RandomErasing()
                                                         ]))
    test_data = dset.ImageFolder('/share/data/vision-greg/DistortedImageNet/Icons-50',
                                 transform=trn.Compose([trn.Resize((32, 32)), trn.ToTensor()]))
    num_classes = 50

    if args.traditional:
        filtered_imgs = []
        for img in train_data.samples:
            img_name = img[0]
            if '_2' not in img_name:
                filtered_imgs.append(img)

        train_data.samples = filtered_imgs[:]

        filtered_imgs = []
        for img in test_data.samples:
            img_name = img[0]
            if '_2' in img_name:
                filtered_imgs.append(img)

        test_data.samples = filtered_imgs[:]

    elif args.subtype:
        test_subclasses = (
            "small_airplane", "top_with_upwards_arrow_above", "soccer_ball", "duck", "hatching_chick", "crossed_swords",
            "passenger_ship", "ledger", "books", "derelict_house_building", "convenience_store", "rabbit_face",
            "cartwheel_type_6", "mantelpiece_clock", "watch", "sun_behind_cloud_with_rain", "wine_glass",
            "face_throwing_a_kiss", "e_mail_symbol", "family_man_boy", "family_man_girl", "family_man_boy_boy",
            "family_man_girl_boy", "family_man_girl_girl", "monorail", "leopard", "chequered_flag", "tulip",
            "womans_sandal",
            "victory_hand", "womans_hat", "broken_heart", "unified_ideograph_5408", "circled_ideograph_accept",
            "closed_lock_with_key", "open_mailbox_with_lowered_flag", "shark", "military_medal",
            "banknote_with_dollar_sign",
            "monkey", "crescent_moon", "mount_fuji", "mobile_phone_off", "no_smoking_symbol", "glowing_star",
            "evergreen_tree",
            "umbrella_with_rain_drops", "racing_car", "factory_worker", "pencil"
        )

        filtered_imgs = []
        for img in train_data.samples:
            img_name = img[0]
            in_test_subclass = False
            for subclass in test_subclasses:
                if subclass in img_name:
                    in_test_subclass = True
                    break
            if in_test_subclass is False:
                filtered_imgs.append(img)

        train_data.samples = filtered_imgs[:]

        filtered_imgs = []
        for img in test_data.samples:
            img_name = img[0]
            in_test_subclass = False
            for subclass in test_subclasses:
                if subclass in img_name:
                    in_test_subclass = True
                    break
            if in_test_subclass is True:
                filtered_imgs.append(img)

        test_data.samples = filtered_imgs[:]

    else:
        filtered_imgs = []
        for img in train_data.samples:
            img_name = img[0]
            if args.test_style not in img_name:
                filtered_imgs.append(img)

        train_data.samples = filtered_imgs[:]

        filtered_imgs = []
        for img in test_data.samples:
            img_name = img[0]
            if args.test_style in img_name:
                filtered_imgs.append(img)

        test_data.samples = filtered_imgs[:]

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# /////////////// Model Setup ///////////////

# Create model

if args.model == 'resnext':
    from models.resnext import ResNeXt
    net = ResNeXt({'input_shape': (1,3,32,32), 'n_classes': num_classes,
                   'base_channels': 32, 'depth': 29, 'cardinality': 8})
if 'shake' in args.model:
    from models.shake_shake import ResNeXt
    net = ResNeXt({'input_shape': (1,3,32,32), 'n_classes': num_classes,
                   'base_channels': 96, 'depth': 26, "shake_forward": True,
                   "shake_backward": True, "shake_image": True})
    args.epochs = 500
    print('Overwriting epochs parameter; now the value is', args.epochs)
elif args.model == 'wrn' or 'wide' in args.model:
    from models.wrn import WideResNet
    net = WideResNet(16, num_classes, 4, dropRate=0.3)
    # args.decay = 5e-4
    # print('Overwriting decay parameter; now the value is', args.decay)
elif args.model == 'resnet':
    from models.resnet import ResNet
    net = ResNet({'input_shape': (1,3,32,32), 'n_classes': num_classes,
                  'base_channels': 16, 'block_type': 'basic', 'depth': 20})
elif args.model == 'densenet':
    from models.densenet import DenseNet
    net = DenseNet({'input_shape': (1,3,32,32), 'n_classes': num_classes,
                    "depth": 40, "block_type": "bottleneck", "growth_rate": 24,
                    "drop_rate": 0.0, "compression_rate": 1})   # 1 is turns compression off

start_epoch = 0
# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset + '_epoch_' + str(i) + '.pt')
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

optimizer = torch.optim.SGD(net.parameters(),
            state['learning_rate'], momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////

from tqdm import tqdm


def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.requires_grad_().cuda(), target.requires_grad_().cuda()

        # forward
        x = net(data)

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss.data) * 0.2

    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()


# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

experiment_indicator = ''
if args.traditional:
    experiment_indicator = '_tradition'
elif args.c100:
    experiment_indicator = '_c100'
elif args.subtype:
    experiment_indicator = '_subtype'

with open(os.path.join(args.save, args.model + experiment_indicator + '_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')


print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # # Save model
    # torch.save(net.state_dict(),
    #            os.path.join(args.save, args.dataset + '_epoch_' + str(epoch) + '.pt'))
    # # Let us not waste space and delete the previous model
    # try: os.remove(os.path.join(args.save, args.dataset + '_epoch_' + str(epoch - 1) + '.pt'))
    # except: True

    # Show results

    with open(os.path.join(args.save, args.model + experiment_indicator + '_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f,\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
              (epoch + 1),
              int(time.time() - begin_epoch),
              state['train_loss'],
              state['test_loss'],
              100 - 100. * state['test_accuracy'])
          )

