# ImageNet-C Corruption Functions

With this package, it is possible to corrupt an image with ImageNet-C corruptions.
These functions are exposed with the function ```corrupt```.

Try
```
from imagenet_c import corrupt

corrupt(<image>, corruption_number=0)
```

The ```corrupt``` function looks like
```
def corrupt(x, severity=1, corruption_name=None, corruption_number=-1):
    """
    :param x: image to corrupt; a 224x224x3 numpy array in [0, 255]
    :param severity: strength with which to corrupt x; an integer in [0, 5]
    :param corruption_name: specifies which corruption function to call;
    must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                    the last four are validation functions
    :param corruption_number: the position of the corruption_name in the above list;
    an integer in [0, 18]; useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    :return: the image x corrupted by a corruption function at the given severity; same shape as input
    """
    ...

```

The "frost" corruption only works should the package be [installed from source](https://github.com/hendrycks/robustness/issues/4#issuecomment-427226016). 
