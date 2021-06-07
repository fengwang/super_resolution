import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from instance_normalization import InstanceNormalization # used in the model
from keras_utils import write_model, read_model
import numpy as np
import imageio


# handling input image, gray -> rgb, rgba -> rgb
def clean_input( image ):
    if len(image.shape) == 3:
        if image.shape[-1] == 3: # direct return RGB image
            return image
        if image.shape[-1] == 4: # RGBA -> RGB
            return image[:,:,:3]

    if len(image.shape) == 2: # GRAY -> RGB
        ans = np.zeros( image.shape + 3 )
        ans[:,:,0], ans[:,:,1], ans[:,:,2] = image, image, image
        return ans

    assert False, f"Unknown image with shape {image.shape}"

model = None

def upsampling_4x( input_low_resolution_image_path, output_4x_high_resolution_image_path=None ):
    # read low resolution image
    im = imageio.imread( input_low_resolution_image_path )
    im = clean_input( im )

    # preparing input for the neural network
    row, col, _ = im.shape
    im = np.asarray( im, dtype='float32' )
    im = im / 127.5 - 1.0
    im = im.reshape( (1,)+im.shape )

    # prepare neural network
    global model
    if model is None:
        script_folder = os.path.dirname(os.path.realpath(__file__))
        model = read_model( f'{script_folder}/model' )

    # predict high resolution image
    him =  model.predict( im, batch_size=1 )
    him = him * 0.5 + 0.5
    him = np.asarray( np.squeeze( him ) * 255, dtype='uint8' )

    # save high resolution image
    if output_4x_high_resolution_image_path is not None:
        imageio.imwrite( output_4x_high_resolution_image_path, him )

    # return high resolution image
    return him

if __name__ == '__main__':
    script_folder = os.path.dirname(os.path.realpath(__file__))
    upsampling_4x( f'{script_folder}/assets/small.png', f'{script_folder}/assets/large.png' )



