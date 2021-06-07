# Super Resolution


![](./assets/demo.png)


Image Super-Resolution using Deep Convolutional Neural Networks.

## Installing

Install and update using [pip](https://pip.pypa.io/en/stable/quickstart/):

```bash
pip3 install super_resolution
```
## Usage

Command line:

```bash
super_resolution INPUT_IMAGE_PATH OUTPUT_IMAGE_PATH
```

```python
# uncomment the follow three lines if you have a Nvidia GPU but you do not want to enable it.
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=''

from super_resolution import cartoon_upsampling_4x
import imageio

cartoon_upsampling_4x( imageio.imread( './a_tiny_image.png', './a_4x_larger_image.png' ) )
```

## Details

+ The super resolution model is inherited from `Ledig C, Theis L, Huszár F, et al. Photo-realistic single image super-resolution using a generative adversarial network, Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 4681-4690.`
+ The training images are downloaded from [Konachan (__NSFW__)](https://konachan.com/).

## License

+ BSD

