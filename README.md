# Benchmarking Neural Network Robustness to Common Corruptions and Perturbations

This repository contains the datasets and some code for the paper [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261) (ICLR 2019) by Dan Hendrycks and Thomas Dietterich.

Requires Python 3+ and PyTorch 0.3+. For evaluation, please download the data from the links below.

## ImageNet-C

<img align="center" src="assets/imagenet-c.png" width="750">

[Download ImageNet-C here.](https://drive.google.com/drive/folders/1HDVw6CmX3HiG0ODFtI75iIfBDxSiSz2K?usp=sharing) [(Mirror.)](https://zenodo.org/record/2235448)

[Download Tiny ImageNet-C here.](https://berkeley.box.com/s/6zt1qzwm34hgdzcvi45svsb10zspop8a) [(Mirror.)](https://zenodo.org/record/2536630)

Tiny ImageNet-C has 200 classes with images of size 64x64, while ImageNet-C has all 1000 classes where each image is the standard size. For even quicker experimentation, there is [CIFAR-10-C](https://zenodo.org/record/2535967), but improvements on CIFAR-10-C may be much less indicative of ImageNet-C improvements. Evaluation using the JPEGs above is strongly prefered to computing the corruptions in memory, so that evaluation is deterministic and consistent.

## ImageNet-C Leaderboard

ImageNet-C Robustness with a ResNet-50 Backbone

|                Method               |                              Reference                             |   mCE   |
|-------------------------------------|--------------------------------------------------------------------|:-------:|
| Stylized ImageNet Data Augmentation | [Geirhos et al.](https://arxiv.org/pdf/1811.12231.pdf) (ICLR 2019) | 69.3%   |
| ResNet-50 Baseline                  |                                                                    | 76.7%   |

Other backbones can obtain better results. For example, a vanilla ResNeXt-101 has an mCE of 62.2%.

Submit a pull request if you beat the state-of-the-art on ImageNet-C.

## ImageNet-P

<img align="center" src="assets/translate.gif" width="224"> <img align="center" src="assets/tilt.gif" width="224"> <img align="center" src="assets/spatter.gif" width="224">

<sub><sup>ImageNet-P sequences are MP4s not GIFs. The spatter perturbation sequence is a validation sequence.</sup></sub>

[Download Tiny ImageNet-P here.](https://berkeley.box.com/s/19m2ppji0xsqgtkrs95329bqftbvncx9) [(Mirror.)](https://zenodo.org/record/2536630)

[Download ImageNet-P here.](https://drive.google.com/drive/folders/1vRrDaWA6-_GaUZqOmovWrr4W34aiSLu7?usp=sharing)

## ImageNet-P Leaderboard

ImageNet-P Perturbation Robustness with a ResNet-50 Backbone

|                Method               |                              Reference                             |   mFR   |   mT5D   |
|-------------------------------------|--------------------------------------------------------------------|:-------:|:-------:|
| Low Pass Filter Pooling (bin-5)     | [Zhang](https://arxiv.org/abs/1904.11486) (ICML 2019)              | 51.2%   | 71.9%   |
| ResNet-50 Baseline                  |                                                                    | 58.0%   | 78.4%   |

Submit a pull request if you beat the state-of-the-art on ImageNet-P.

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2019robustness,
      title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
      author={Dan Hendrycks and Thomas Dietterich},
      journal={Proceedings of the International Conference on Learning Representations},
      year={2019}
    }

Part of the code was contributed by [Tom Brown](https://github.com/nottombrown).

## Icons-50 (From an Older Draft)

<img align="center" src="assets/icons-50.png" width="500">

Download Icons-50 [here](https://berkeley.box.com/s/jcem6ik7rxr6594lg99kmrdo01ue6yjt) or [here.](https://drive.google.com/drive/folders/16_kaFo3uUoS-U8FTDm4nUh6Vo21UVnJX?usp=sharing)

