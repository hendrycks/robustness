import os
import numpy as np
import torch
import torchvision.transforms as trn
import torchvision.transforms.functional as trn_F
import cv2
from PIL import Image as PILImage
import skimage.color as skcolor
from skimage.util import random_noise
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from io import BytesIO
import ctypes
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from tempfile import gettempdir
from shutil import rmtree
import torchvision.datasets as dset

# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def brightness(_x, c=0.):
    _x = np.array(_x, copy=True) / 255.
    _x = skcolor.rgb2hsv(_x)
    _x[:, :, 2] = np.clip(_x[:, :, 2] + c, 0, 1)
    _x = skcolor.hsv2rgb(_x)

    return np.uint8(_x * 255)


def to_numpy(image):
    return np.uint8(image.numpy().transpose(1,2,0) * 255)


test_data = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=False)

cifar_p = []
labels = []

for img, label in zip(test_data.data, test_data.targets):
    seq = []
    labels.append(label)

    # /////////////// Test Data ///////////////

    # /////////////// Gaussian Noise Code ///////////////

    x = trn.ToTensor()(img)
    seq.append(img)

    for i in range(1, 31):
        z = to_numpy(torch.clamp(x + 0.02 * torch.randn_like(x), 0, 1))
        seq.append(z)

    # for i in range(1, 31):
    #     z = torch.clamp(x + 0.04 * torch.randn_like(x), 0, 1)
    #     seq.append(to_numpy(z))

    # for i in range(1, 31):
    #     z = torch.clamp(x + 0.06 * torch.randn_like(x), 0, 1)
    #     seq.append(to_numpy(z))

    # /////////////// End Gaussian Noise Code ///////////////

    # /////////////// Shot Noise Code ///////////////

    # x = img
    # seq.append(img)

    # for i in range(1, 31):
    #     z = np.array(x, copy=True) / 255.
    #     z = np.uint8(255 * np.clip(np.random.poisson(z * 700) / 700., 0, 1))
    #     seq.append(z)

    # for i in range(1, 31):
    #     z = np.array(x, copy=True) / 255.
    #     z = np.uint8(255 * np.clip(np.random.poisson(z * 400) / 400., 0, 1))
    #     seq.append(z)

    # for i in range(1, 31):
    #     z = np.array(x, copy=True) / 255.
    #     z = np.uint8(255 * np.clip(np.random.poisson(z * 200) / 200., 0, 1))
    #     seq.append(z)

    # /////////////// End Shot Noise Code ///////////////

    # /////////////// Motion Blur Code ///////////////

    # for i in range(0, 21):
    #     z = PILImage.fromarray(img)
    #     output = BytesIO()
    #     z.save(output, format='PNG')
    #     z = MotionImage(blob=output.getvalue())
    #
    #     z.motion_blur(radius=6, sigma=1.8, angle=(i - 20) * 9)
    #
    #     z = cv2.imdecode(np.fromstring(z.make_blob(), np.uint8),
    #                      cv2.IMREAD_UNCHANGED)
    #
    #     if z.shape != (32, 32):
    #         z = np.clip(z[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    #     else:  # grayscale to RGB
    #         z = np.clip(np.array([z, z, z]).transpose((1, 2, 0)), 0, 255)
    #
    #     seq.append(np.uint8(z))

    # /////////////// End Motion Blur Code ///////////////

    # /////////////// Zoom Blur Code ///////////////

    # seq.append(img)
    # avg = trn.ToTensor()(img)
    #
    # for i in range(1, 31):
    #     z = trn.CenterCrop(32)(trn_F.affine(PILImage.fromarray(img), angle=0, translate=(0, 0),
    #                                         scale=1+0.004*i, shear=0, resample=PILImage.BILINEAR))
    #     avg += trn.ToTensor()(z)
    #     seq.append(np.array(trn.ToPILImage()(avg / (i + 1))))

    # /////////////// End Zoom Blur Code ///////////////

    # /////////////// Snow Code ///////////////

    # x = np.array(img) / 255.
    #
    # snow_layer = np.random.normal(size=(32, 32), loc=0.05, scale=0.3)
    #
    # snow_layer = clipped_zoom(snow_layer[..., np.newaxis], 2)
    # snow_layer[snow_layer < 0.5] = 0
    #
    # snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    # output = BytesIO()
    # snow_layer.save(output, format='PNG')
    # output = output.getvalue()
    #
    # for i in range(0, 31):
    #     moving_snow = MotionImage(blob=output)
    #     moving_snow.motion_blur(radius=10, sigma=2, angle=i*4-150)
    #
    #     snow_layer = cv2.imdecode(np.fromstring(moving_snow.make_blob(), np.uint8),
    #                               cv2.IMREAD_UNCHANGED) / 255.
    #     snow_layer = snow_layer[..., np.newaxis]
    #
    #     z = 0.85 * x + (1 - 0.85) * np.maximum(
    #         x, cv2.cvtColor(np.float32(x), cv2.COLOR_RGB2GRAY).reshape(32, 32, 1) * 1.5 + 0.5)
    #
    #     z = np.uint8(np.clip(z + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255)
    #
    #     seq.append(z)

    # /////////////// End Snow Code ///////////////

    # /////////////// Brightness Code ///////////////

    # x = PILImage.fromarray(img)
    #
    # for i in range(0, 31):
    #     z = brightness(x, c=(i - 15) * 1.5 / 100.)
    #     seq.append(z)

    # /////////////// End Brightness Code ///////////////

    # /////////////// Translate Code ///////////////

    # x = PILImage.fromarray(img)
    #
    # for i in range(0,16):
    #     z = trn_F.affine(x, angle=0, translate=(i-7, 0), scale=1, shear=0)
    #     seq.append(np.array(z))

    # /////////////// End Translate Code ///////////////

    # /////////////// Rotate Code ///////////////

    # x = PILImage.fromarray(img)
    #
    # for i in range(0, 31):
    #     z = trn_F.affine(x, angle=i-15, translate=(0, 0),
    #                      scale=1., shear=0, resample=PILImage.BILINEAR)
    #     seq.append(np.array(z))

    # /////////////// End Rotate Code ///////////////

    # /////////////// Tilt Code ///////////////

    # x = np.array(img)
    # h, w = x.shape[0:2]
    #
    # for i in range(0, 31):
    #     phi, theta = np.deg2rad(0.75*(i-15)), np.deg2rad(0.75*(i-15))
    #
    #     f = np.sqrt(w ** 2 + h ** 2)
    #
    #     P1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])
    #
    #     RX = np.array([[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0],
    #                    [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])
    #
    #     RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0], [0, 1, 0, 0],
    #                    [np.sin(phi), 0, np.cos(phi), 0], [0, 0, 0, 1]])
    #
    #     T = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
    #                   [0, 0, 1, f], [0, 0, 0, 1]])
    #
    #     P2 = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])
    #
    #     mat = P2 @ T @ RX @ RY @ P1
    #
    #     z = cv2.warpPerspective(x, mat, (w, h))
    #     seq.append(z)

    # /////////////// End Tilt Code ///////////////

    # /////////////// Scale Code ///////////////

    # x = PILImage.fromarray(img)
    #
    # for i in range(0, 31):
    #     z = trn.CenterCrop(32)(trn_F.affine(x, angle=0, translate=(0, 0),
    #                                         scale=(i * 2 + 70) / 100., shear=0, resample=PILImage.BILINEAR))
    #     seq.append(np.array(z))

    # /////////////// End Scale Code ///////////////

    # /////////////// Validation Data ///////////////

    # /////////////// Speckle Noise Code ///////////////

    # seq.append(img)
    # x = np.array(img) / 255.

    # for i in range(1, 31):
    #     z = np.uint8(255 * np.clip(x + x * np.random.normal(size=x.shape, scale=0.05), 0, 1))
    #     seq.append(z)

    # for i in range(1, 31):
    #     z = np.uint8(255 * np.clip(x + x * np.random.normal(size=x.shape, scale=0.07), 0, 1))
    #     seq.append(z)

    # for i in range(1, 31):
    #     z = np.uint8(255 * np.clip(x + x * np.random.normal(size=x.shape, scale=0.1), 0, 1))
    #     seq.append(z)

    # /////////////// End Speckle Noise Code ///////////////

    # /////////////// Gaussian Blur Code ///////////////

    # for i in range(0, 31):
    #     z = np.uint8(255*gaussian(np.array(img, copy=True)/255.,
    #                               sigma=0.3 + 0.015*i, multichannel=True, truncate=7.0))
    #     seq.append(z)

    # /////////////// End Gaussian Blur Code ///////////////

    # /////////////// Spatter Code ///////////////

    # x = cv2.cvtColor(np.array(img, dtype=np.float32) / 255., cv2.COLOR_BGR2BGRA)
    #
    # liquid_layer = np.random.normal(size=x.shape[:2], loc=0.6, scale=0.265)
    # liquid_layer = gaussian(liquid_layer, sigma=1.75)
    # liquid_layer[liquid_layer < 0.7] = 0
    #
    # for i in range(0, 31):
    #
    #     liquid_layer_i = (liquid_layer * 255).astype(np.uint8)
    #     dist = 255 - cv2.Canny(liquid_layer_i, 50, 150)
    #     dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
    #     _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
    #     dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
    #     dist = cv2.equalizeHist(dist)
    #     ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    #     dist = cv2.filter2D(dist, cv2.CV_8U, ker)
    #     dist = cv2.blur(dist, (3, 3)).astype(np.float32)
    #
    #     m = cv2.cvtColor(liquid_layer_i * dist, cv2.COLOR_GRAY2BGRA)
    #     m /= np.max(m, axis=(0, 1))
    #     m *= 0.6
    #
    #     # water is pale turqouise
    #     color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
    #                             238 / 255. * np.ones_like(m[..., :1]),
    #                             238 / 255. * np.ones_like(m[..., :1])), axis=2)
    #
    #     color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
    #
    #     z = np.uint8(cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255)
    #
    #     liquid_layer = np.apply_along_axis(lambda mat:
    #                                        np.convolve(mat, np.array([0.2, 0.8]), mode='same'),
    #                                        axis=0, arr=liquid_layer)
    #
    #     seq.append(z)

    # /////////////// End Spatter Code ///////////////

    # /////////////// Shear Code ///////////////

    # for i in range(0, 31):
    #     z = trn.CenterCrop(32)(trn_F.affine(PILImage.fromarray(img), angle=0, translate=(0, 0),
    #                                         scale=1., shear=i-15, resample=PILImage.BILINEAR))
    #     seq.append(np.array(z))

    # /////////////// End Shear Code ///////////////

    cifar_p.append(seq)


np.save('/share/data/vision-greg2/users/dan/datasets/CIFAR-10-P/gaussian_noise.npy',
        np.array(cifar_p).astype(np.uint8))
