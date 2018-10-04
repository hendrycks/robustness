import setuptools
from glob import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imagenet_c",
    version="0.0.2",
    author="Dan Hendrycks",
    author_email="hendrycks@berkeley.edu",
    description="Access to ImageNet-C corruption functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hendrycks/robustness/tree/master/ImageNet-C/imagenet_c",
    packages=setuptools.find_packages(),
    package_data={
        "imagenet_c.frost": [
            "frost1.png", "frost2.png", "frost3.png",
            "frost4.jpg", "frost5.jpg", "frost6.jpg"
        ],
    },
    install_requires=[
        'wand ~= 0.4',
        'opencv-python ~= 3.4',
    ],
    #include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

