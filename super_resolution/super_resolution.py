
__all__ = ['cartoon_upsampling_4x', 'cartoon_upsampling_8x']

import os
import numpy as np
import imageio
import os.path

#from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.keras import backend as K
from requests import get

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# download model files from github release
def download_remote_model( model_name, model_url):
    user_model_path = os.path.join( os.path.expanduser('~'), '.deepoffice', 'super_resolution', 'model' )
    model_path = os.path.join( user_model_path, model_name )

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_name = model_url.rsplit('/', 1)[1]
    local_model_path = os.path.join( model_path, file_name )
    if not os.path.isfile(local_model_path):
        print( f'downloading model file {local_model_path} from {model_url}' )
        with open(local_model_path, "wb") as file:
            response = get(model_url)
            file.write(response.content)
        print( f'downloaded model file {local_model_path} from {model_url}' )

    return local_model_path

def download_4x_model():
    model_name = 'super_resolution_4x'
    download_remote_model( model_name, 'https://github.com/fengwang/super_resolution/releases/download/0.3/js.json' )
    download_remote_model( model_name, 'https://github.com/fengwang/super_resolution/releases/download/0.3/weights.h5' )
    return os.path.join( os.path.expanduser('~'), '.deepoffice', 'super_resolution', 'model', model_name )

def download_8x_model():
    model_name = 'super_resolution_8x'
    download_remote_model( model_name, 'https://github.com/fengwang/super_resolution/releases/download/0.3.1/js.json' )
    download_remote_model( model_name, 'https://github.com/fengwang/super_resolution/releases/download/0.3.1/weights.h5' )
    return os.path.join( os.path.expanduser('~'), '.deepoffice', 'super_resolution', 'model', model_name )


def download_denoised_8x_model():
    model_name = 'denoised_super_resolution_8x'
    download_remote_model( model_name, 'https://github.com/fengwang/super_resolution/releases/download/0.3.2/js.json' )
    download_remote_model( model_name, 'https://github.com/fengwang/super_resolution/releases/download/0.3.2/weights.h5' )
    return os.path.join( os.path.expanduser('~'), '.deepoffice', 'super_resolution', 'model', model_name )





# credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
class InstanceNormalization(Layer):
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

get_custom_objects().update({'InstanceNormalization': InstanceNormalization})

'''
    Example:
        model = read_model( './cached_folder' )
'''
def read_model(directory):
    #weights_path = f'{directory}/weights.h5'
    weights_path = os.path.join( directory, 'weights.h5' )
    if not os.path.isfile(weights_path):
        print( f'Failed to find weights from file {weights_path}' )
        return None

    #json_path = f'{directory}/js.json'
    json_path = os.path.join( directory, 'js.json' )
    if not os.path.isfile(json_path):
        print( f'Failed to find model from file {json_path}' )
        return None

    js_file = open( json_path, 'r' )
    model_json = js_file.read()
    js_file.close()
    model = model_from_json( model_json, custom_objects={"InstanceNormalization": InstanceNormalization} )
    model.load_weights( weights_path )
    return model


def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )



# handling input image, gray -> rgb, rgba -> rgb
def clean_input( image ):
    if len(image.shape) == 3:
        if image.shape[-1] == 3: # direct return RGB image
            return image
        if image.shape[-1] == 4: # RGBA -> RGB
            return rgba2rgb( image )
            #return image[:,:,:3]

    if len(image.shape) == 2: # GRAY -> RGB
        ans = np.zeros( image.shape + 3 )
        ans[:,:,0], ans[:,:,1], ans[:,:,2] = image, image, image
        return ans

    assert False, f"Unknown image with shape {image.shape}"

cartoon_model = None
def cartoon_upsampling_4x( input_low_resolution_image_path, output_4x_high_resolution_image_path=None ):
    # read low resolution image
    im = imageio.imread( input_low_resolution_image_path )
    im = clean_input( im )

    # preparing input for the neural network
    row, col, _ = im.shape
    im = np.asarray( im, dtype='float32' )
    im = im / 127.5 - 1.0
    im = im.reshape( (1,)+im.shape )

    # prepare neural network
    global cartoon_model
    if cartoon_model is None:
        script_folder = os.path.dirname(os.path.realpath(__file__))
        model_path = download_4x_model()
        cartoon_model = read_model( model_path )

    # predict high resolution image
    ans = cartoon_model.predict( im, batch_size=1 )
    ans = ans * 0.5 + 0.5
    ans = np.asarray( np.squeeze( ans ) * 255, dtype='uint8' )

    # save high resolution image
    if output_4x_high_resolution_image_path is not None:
        imageio.imwrite( output_4x_high_resolution_image_path, ans )

    # return high resolution image
    return ans


cartoon_model_8x = None
def cartoon_upsampling_8x( input_low_resolution_image_path, output_8x_high_resolution_image_path=None ):
    # read low resolution image
    im = imageio.imread( input_low_resolution_image_path )
    im = clean_input( im )

    # preparing input for the neural network
    row, col, _ = im.shape
    im = np.asarray( im, dtype='float32' )
    im = im / 127.5 - 1.0
    im = im.reshape( (1,)+im.shape )

    # prepare neural network
    global cartoon_model_8x
    if cartoon_model_8x is None:
        script_folder = os.path.dirname(os.path.realpath(__file__))
        #model_path = os.path.join(script_folder, 'resources', 'cartoon_model_8x' )
        #model_path = download_8x_model()
        model_path = download_denoised_8x_model()
        cartoon_model_8x = read_model( model_path )

    # predict high resolution image
    ans = cartoon_model_8x.predict( im, batch_size=1 )
    ans = ans * 0.5 + 0.5
    ans = np.asarray( np.squeeze( ans ) * 255, dtype='uint8' )

    # save high resolution image
    if output_8x_high_resolution_image_path is not None:
        imageio.imwrite( output_8x_high_resolution_image_path, ans )

    # return high resolution image
    return ans




