# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L

activations = {
	"sigmoid": F.sigmoid, 
	"tanh": F.tanh, 
	"softplus": F.softplus, 
	"relu": F.relu, 
	"leaky_relu": F.leaky_relu, 
	"elu": F.elu
}

class Params():
	def __init__(self, dict=None):
		self.audio_channels = 256

		# entire architecture
		# causal conv stack -> residual dilated conv stack -> skip-connections conv -> softmax

		self.causal_conv_no_bias = True
		self.causal_conv_kernel_size = 2
		self.causal_conv_apply_batchnorm = True
		self.causal_conv_batchnorm_before_conv = True
		# [<- input   output ->]
		# audio input -> conv -> (128,) -> conv -> (128,) -> conv -> (128,) -> residual dilated conv stack
		self.causal_conv_channels = [128, 128, 128]

		self.residual_conv_dilation_no_bias = True
		self.residual_conv_projection_no_bias = True
		self.residual_conv_kernel_size = 2
		self.residual_conv_apply_batchnorm = True
		self.residual_conv_batchnorm_before_conv = True
		# [<- input   output ->]
		# causal conv output (128,) -> conv -> (32,) -> 1x1 conv -> (128,) -> conv -> (32,) -> 1x1 conv -> (128,) -> ...
		self.residual_conv_channels = [32, 32, 32]
		# [<- input   output ->]
		self.residual_conv_dilations = [1, 2, 4, 8, 2, 4, 8, 4, 8]

		self.skip_connections_conv_no_bias = False
		self.skip_connections_conv_kernel_size = 2
		self.skip_connections_conv_apply_batchnorm = True
		self.skip_connections_conv_batchnorm_before_conv = True
		# [<- input   output ->]
		# skip-connections -> ReLU -> conv -> (128,) -> ReLU -> conv -> (256,) -> softmax -> prediction
		self.skip_connections_conv_channels = [128, 256]

		self.gpu_enabled = True
		self.learning_rate = 0.001
		self.gradient_momentum = 0.9
		self.gradient_clipping = 10.0

		if dict:
			self.from_dict(dict)

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			if not hasattr(self, attr):
				raise Exception("invalid parameter '{}'".format(attr))
			setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			dict[attr] = value
		return dict

	def dump(self):
		print "params:"
		for attr, value in self.__dict__.iteritems():
			print "	{}: {}".format(attr, value)

	def check(self):
		base = Params()
		for attr, value in self.__dict__.iteritems():
			if not hasattr(base, attr):
				raise Exception("invalid parameter '{}'".format(attr))
		if self.causal_conv_channels[-1] != self.skip_connections_conv_channels[0]:
			raise Exception("causal_conv_channels[-1] != skip_connections_conv_channels[0]")
		if self.audio_channels != self.skip_connections_conv_channels[-1]:
			raise Exception("audio_channels != skip_connections_conv_channels[-1]")

def sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])
	
class GradientClipping(object):
	name = "GradientClipping"

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(sum_sqnorm([p.grad for p in opt.target.params()]))
		if norm == 0:
			return
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

class Padding1d(function.Function):
	def __init__(self, pad=0):
		self.pad = pad

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 1)
		x_type, = in_types

		type_check.expect(
			x_type.dtype == np.float32,
			x_type.ndim == 4,
		)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		x, = inputs
		n_batch = x.shape[0]
		in_channels = x.shape[1]
		height = x.shape[2]
		width = x.shape[3]
		paded_width = width + self.pad
		out_shape = (n_batch, in_channels, height, paded_width)
		output = xp.zeros(out_shape, dtype=xp.float32)
		output[:,:,:,self.pad:] = x
		return output,

	def backward(self, inputs, grad_outputs):
		return grad_outputs[0][:,:,:,self.pad:],

class Slice1d(function.Function):
	def __init__(self, cut=0):
		self.cut = cut

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 1)
		x_type, = in_types

		type_check.expect(
			x_type.dtype == np.float32,
			x_type.ndim == 4,
		)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		x, = inputs
		width = x.shape[3]
		cut_width = width - self.cut
		output = output[:,:,:,:cut_width]
		return output,

	def backward(self, inputs, grad_outputs):
		xp = cuda.get_array_module(inputs[0])
		return xp.append(grad_outputs[0], xp.zeros((self.cut,), dtype=xp.float32), axis=3)

class DilatedConvolution1D(L.Convolution2D):

	def padding_1d(self, x, pad):
		return Padding1d(pad=pad)(x)

	def slice_1d(self, x, pad):
		return Slice1d(pad=pad)(x)

	def __call__(self, x, dilation=1):
		if self.has_uninitialized_params:
			self._initialize_params(x.shape[1])
		kernel_width = self.ksize[0]

		batchsize = x.shape[0]
		input_x_channel = x.shape[1]
		input_x_width = x.shape[3]

		# padding
		# # of elements in padded x >= input_x_width + dilation * (kernel_width - 1) 
		pad = input_x_width + dilation * (self.kernel_width - 1)
		padded_x_width = input_x_width + pad
		# we need more padding to perform convlution
		if padded_x_width < kernel_width * dilation:
			pad += kernel_width * dilation - padded_x_width
			padded_x_width = input_x_width + pad
		pad += padded_x_width % dilation
		padded_x = self.padding_1d(x, pad)

		# reshape to skip (dilation - 1) elements
		padded_x = F.reshape(batchsize * self.dilation, input_x_channel, 1, -1)

		# convolution
		out = F.convolution_2d(x, self.W, self.b, self.stride, self.pad, self.use_cudnn)

		# Remove padded elements
		cut = padded_x.shape[3] - input_x_width
		out = self.slice_1d(padded_x, cut)

		return out

class ResidualConvLayer(chainer.Chain):
	def __init__(self, **layers):
		super(ResidualConvLayer, self).__init__(**layers)
		self.apply_batchnorm = False
		self.batchnorm_before_conv = True
		self.dilation = 1

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def __call__(self, x, test=False):
		# batchnorm
		if self.batchnorm_before_conv and self.apply_batchnorm:
			x = self.batchnorm(x, test=test)

		# gated activation
		if self.batchnorm_before_conv == False and self.apply_batchnorm:
			z = F.tanh(self.batchnorm_f(self.wf(x, dilation=self.dilation)), test=test) * F.sigmoid(self.batchnorm_g(self.wg(x, dilation=self.dilation), test=test))
		else:
			z = F.tanh(self.wf(x, dilation=self.dilation)) * F.sigmoid(self.wg(x, dilation=self.dilation))

		print x.data.shape
		print z.data.shape

		# 1x1 conv
		z = self.projection(z)
		print z.data.shape

		# residual
		output = z + x
		return output, z

class WaveNet():

	def __init__(self, params):
		params.check()
		self.params = params

		# stack residual blocks
		ksize = (params.residual_conv_kernel_size, 1)
		nobias_dilation = params.residual_conv_dilation_no_bias
		nobias_projection = params.residual_conv_projection_no_bias
		self.residual_conv_layers = []
		channels = [(params.audio_channels, params.residual_conv_channels[0])]
		channels += zip(params.residual_conv_channels[:-1], params.residual_conv_channels[1:])
		for i, (n_in, n_out) in enumerate(channels):
			shape_w = (n_out, n_in, ksize[0], ksize[1])
			initial_w = np.random.normal(scale=math.sqrt(2.0 / (ksize[0] * ksize[0] * n_out)), size=shape_w)
			initial_w = np.ones(shape_w).astype(np.float32)
			attributes = {}
			dilation = params.residual_conv_dilations[i]
			attributes["wf"] = DilatedConvolution1D(n_in, n_out, ksize, stride=1, nobias=nobias_dilation, initialW=initial_w)
			attributes["wg"] = DilatedConvolution1D(n_in, n_out, ksize, stride=1, nobias=nobias_dilation, initialW=initial_w)
			attributes["projection"] = L.Convolution2D(n_out, n_in, 1, stride=1, nobias=nobias_projection)
			if param.residual_conv_batchnorm_before_conv:
				attributes["batchnorm"] = L.BatchNormalization(n_in)
			else:
				attributes["batchnorm_g"] = L.BatchNormalization(n_out)
				attributes["batchnorm_f"] = L.BatchNormalization(n_out)
			conv_layer = ResidualConvLayer(**attributes)
			conv_layer.apply_batchnorm = params.residual_conv_apply_batchnorm
			conv_layer.batchnorm_before_conv = params.residual_conv_batchnorm_before_conv
			conv_layer.dilation = dilation
			conv_layer.kernel_width = ksize[0]
			self.residual_conv_layers.append(conv_layer)

	@property
	def gpu_enabled(self):
		return self.params.gpu_enabled
	

	def forward_residual(self, x_batch):
		x_batch = Variable(x_batch)
		skip_connections = 0
		for layer in self.residual_conv_layers:
			output, z = layer(x_batch)
			print output.data
			print z.data

	def loss(self, x_batch):
		pass

	def save(self, dir="./"):
		pass

	def load(self, dir="./"):
		pass