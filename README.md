# Benchmarking Neural Network Robustness to Common Corruptions and Perturbations

This repository contains the datasets and some code for the paper [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261) (ICLR 2019) by Dan Hendrycks and Thomas Dietterich.

Requires Python 3+ and PyTorch 0.3+. For evaluation, please download the data from the links below.

## ImageNet-C

<img align="center" src="assets/imagenet-c.png" width="750">

[Download ImageNet-C here.](https://zenodo.org/record/2235448) [(Mirror.)](https://drive.google.com/drive/folders/1HDVw6CmX3HiG0ODFtI75iIfBDxSiSz2K?usp=sharing)

[Download Tiny ImageNet-C here.](https://zenodo.org/record/2536630) [(Mirror.)](https://berkeley.box.com/s/6zt1qzwm34hgdzcvi45svsb10zspop8a)

Tiny ImageNet-C has 200 classes with images of size 64x64, while ImageNet-C has all 1000 classes where each image is the standard size. For even quicker experimentation, there is [CIFAR-10-C](https://zenodo.org/record/2535967) and [CIFAR-100-C](https://zenodo.org/record/3555552). Evaluation using the JPEGs above is strongly prefered to computing the corruptions in memory, so that evaluation is deterministic and consistent.

## ImageNet-C Leaderboard

ImageNet-C Robustness with a ResNet-50 Backbone

|                Method               |                              Reference                             |   mCE   |    Clean Error |
|-------------------------------------|--------------------------------------------------------------------|:-------:| :-------:|
| [AugMix](https://github.com/google-research/augmix) | [Hendrycks and Mu et al.](https://arxiv.org/pdf/1912.02781.pdf) (ICLR 2020) | 65.3%   |  22.47%
| Stylized ImageNet Data Augmentation | [Geirhos et al.](https://arxiv.org/pdf/1811.12231.pdf) (ICLR 2019) | 69.3%   |  25.41%
| Patch Uniform | [Lopes et al.](https://arxiv.org/abs/1906.02611)  | 74.3%   |  24.5%
| ResNet-50 Baseline                  |                                                                    | 76.7%   | 23.85%

Other backbones can obtain better results. For example, a vanilla ResNeXt-101 has an mCE of 62.2%.
Note Lopes et al. have a ResNet-50 backbone with an mCE of [80.6](https://openreview.net/pdf?id=S1gmrxHFvB#page=7&zoom=100,144,580), so their improvement is larger than what is immediately suggested by the table.

Submit a pull request if you beat the state-of-the-art with a ResNet-50 backbone on ImageNet-C.

## AlexNet ImageNet-C Error

Use these values to normalize raw corruption error to calculate mCE:

|    Corruption    | Average    | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
|------------------|------------|------------|------------|------------|------------|------------|
| Gaussian Noise   | 0.886428   | 0.69528    | 0.82542    | 0.93554    | 0.98138    | 0.99452    |
| Shot Noise       | 0.894468   | 0.71224    | 0.85108    | 0.93574    | 0.98182    | 0.99146    |
| Impulse Noise    | 0.922640   | 0.78374    | 0.89808    | 0.94870    | 0.98720    | 0.99548    |
| Defocus Blur     | 0.819880   | 0.65624    | 0.73202    | 0.85036    | 0.91364    | 0.94714    |
| Glass Blur       | 0.826268   | 0.64308    | 0.75054    | 0.88806    | 0.91622    | 0.93344    |
| Motion Blur      | 0.785948   | 0.58430    | 0.70048    | 0.82108    | 0.89750    | 0.92638    |
| Zoom Blur        | 0.798360   | 0.70008    | 0.76992    | 0.80784    | 0.84198    | 0.87198    |
| Snow             | 0.866816   | 0.71726    | 0.88392    | 0.86468    | 0.91870    | 0.94952    |
| Frost            | 0.826572   | 0.61390    | 0.79734    | 0.88790    | 0.89942    | 0.93430    |
| Fog              | 0.819324   | 0.67474    | 0.76050    | 0.84378    | 0.87260    | 0.94500    |
| Brightness       | 0.564592   | 0.45140    | 0.48502    | 0.54048    | 0.62166    | 0.72440    |
| Contrast         | 0.853204   | 0.64548    | 0.76150    | 0.88874    | 0.97760    | 0.99270    |
| Elastic          | 0.646056   | 0.52596    | 0.70116    | 0.55686    | 0.64076    | 0.80554    |
| Pixelate         | 0.717840   | 0.52218    | 0.54620    | 0.73728    | 0.87092    | 0.91262    |
| JPEG Compression | 0.606500   | 0.51002    | 0.54718    | 0.57294    | 0.65458    | 0.74778    |
| Speckle Noise    | 0.845388   | 0.66192    | 0.74440    | 0.90246    | 0.94548    | 0.97268    |
| Gaussian Blur    | 0.787108   | 0.54732    | 0.70444    | 0.82574    | 0.89864    | 0.95940    |
| Spatter          | 0.717512   | 0.47196    | 0.62194    | 0.75052    | 0.84132    | 0.90182    |
| Saturate         | 0.658248   | 0.59342    | 0.65514    | 0.51174    | 0.70834    | 0.82260    |

## ImageNet-P

<img align="center" src="assets/translate.gif" width="224"> <img align="center" src="assets/tilt.gif" width="224"> <img align="center" src="assets/spatter.gif" width="224">

<sub><sup>ImageNet-P sequences are MP4s not GIFs. The spatter perturbation sequence is a validation sequence.</sup></sub>

[Download Tiny ImageNet-P here.](https://zenodo.org/record/2536630) [(Mirror.)](https://berkeley.box.com/s/19m2ppji0xsqgtkrs95329bqftbvncx9)

[Download ImageNet-P here.](https://zenodo.org/record/3565846) [(Mirror.)](https://drive.google.com/drive/folders/1vRrDaWA6-_GaUZqOmovWrr4W34aiSLu7?usp=sharing)

## ImageNet-P Leaderboard

ImageNet-P Perturbation Robustness with a ResNet-50 Backbone

|                Method               |                              Reference                             |   mFR   |   mT5D   |
|-------------------------------------|--------------------------------------------------------------------|:-------:|:-------:|
| [AugMix](https://github.com/google-research/augmix) | [Hendrycks and Mu et al.](https://arxiv.org/pdf/1912.02781.pdf) (ICLR 2020)              |   37.4%  |    |
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

