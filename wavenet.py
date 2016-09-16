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

		self.residual_conv_no_bias = True
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

class ResidualConvLayer(chainer.Chain):
	def __init__(self, **layers):
		super(ResidualConvLayer, self).__init__(**layers)
		self.apply_batchnorm = False
		self.test = False

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def __call__(self, x, test=False, apply_f=True):
		if self.apply_batchnorm:
			x = self.batchnorm(x)
		# gated activation
		z = F.tanh(self.wf(x)) * F.sigmoid(self.wg(x))
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
		nobias = params.residual_conv_no_bias
		self.residual_conv_layers = []
		channels = [(params.audio_channels, params.residual_conv_channels[0])]
		channels += zip(params.residual_conv_channels[:-1], params.residual_conv_channels[1:])
		for i, (n_in, n_out) in enumerate(channels):
			w_shape = (n_out, n_in, ksize[0], ksize[1])
			initial_w = np.ones(w_shape).astype(np.float32)
			attributes = {}
			attributes["wf"] = L.Convolution2D(n_in, n_out, ksize, stride=ksize, nobias=nobias, initialW=initial_w)
			attributes["wg"] = L.Convolution2D(n_in, n_out, ksize, stride=ksize, nobias=nobias, initialW=initial_w)
			attributes["projection"] = L.Convolution2D(n_out, n_in, 1, stride=1)
			attributes["batchnorm"] = L.BatchNormalization(n_in)
			conv_layer = ResidualConvLayer(**attributes)
			conv_layer.apply_batchnorm = params.apply_batchnorm
			self.residual_conv_layers.append(conv_layer)

	@property
	def input_receptive_size(self):
		params = self.params
		return params.residual_conv_kernel_size ** (len(params.residual_conv_channels) + 1)

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