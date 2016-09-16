from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from args import args
from model import params, wavenet


input_size = 4
batchsize = 2
data = np.ones((batchsize, params.audio_channels, input_size, 1)).astype(np.float32)
print data.shape
print input_size
wavenet.forward_residual(data)

shape_w = (n_out, n_in, ksize[0], ksize[1])
initial_w = np.random.normal(scale=math.sqrt(2 / (ksize * ksize * n_out)), size=shape_w)
attributes = {}
dilation = params.residual_conv_dilations[i]
attributes["wf"] = L.Convolution2D(n_in, n_out, ksize, stride=1, nobias=nobias_dilation, initialW=initial_w)
attributes["wg"] = L.Convolution2D(n_in, n_out, ksize, stride=1, nobias=nobias_dilation, initialW=initial_w)
attributes["projection"] = L.Convolution2D(n_out, n_in, 1, stride=1, nobias=nobias_projection)
attributes["batchnorm"] = L.BatchNormalization(n_in)
conv_layer = ResidualConvLayer(**attributes)
conv_layer.apply_batchnorm = params.residual_conv_apply_batchnorm
conv_layer.batchnorm_before_conv = params.residual_conv_batchnorm_before_conv
conv_layer.dilation = dilation
self.residual_conv_layers.append(conv_layer)