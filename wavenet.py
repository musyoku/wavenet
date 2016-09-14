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

class Conf():
	def __init__(self):
		self.audio_channels = 256

		self.residual_conv_kernel_size = 2

		# [<- input   output ->]
		# audio input -> conv -> (256,) -> conv -> (256,) -> conv -> (256,) 
		# to stack more layer, do [256, 256, 256, 256, 256, 256, ...]
		self.residual_conv_channels= [256, 256, 256]
		self.residual_conv_no_bias = True

		# [<- input   output ->]
		# skip-connections -> ReLu -> conv -> (256,) -> ReLU -> conv -> (256,) -> softmax -> prediction
		self.skip_connections_conv_channels = [256, 256]

		self.apply_batchnorm = False
		self.gpu_enabled = True
		self.learning_rate = 0.0003
		self.gradient_momentum = 0.9
		self.gradient_clipping = 10.0

	def check(self):
		base = Conf()
		for attr, value in self.__dict__.iteritems():
			if not hasattr(base, attr):
				raise Exception("invalid parameter '{}'".format(attr))

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

	def __init__(self, conf):
		conf.check()
		self.conf = conf

		# stack residual blocks
		ksize = (conf.residual_conv_kernel_size, 1)
		nobias = conf.residual_conv_no_bias
		self.residual_conv_layers = []
		channels = [(conf.audio_channels, conf.residual_conv_channels[0])]
		channels += zip(conf.residual_conv_channels[:-1], conf.residual_conv_channels[1:])
		for i, (n_in, n_out) in enumerate(channels):
			w_shape = (n_out, n_in, ksize[0], ksize[1])
			initial_w = np.ones(w_shape).astype(np.float32)
			attributes = {}
			attributes["wf"] = L.Convolution2D(n_in, n_out, ksize, stride=ksize, nobias=nobias, initialW=initial_w)
			attributes["wg"] = L.Convolution2D(n_in, n_out, ksize, stride=ksize, nobias=nobias, initialW=initial_w)
			attributes["projection"] = L.Convolution2D(n_out, n_in, 1, stride=1)
			attributes["batchnorm"] = L.BatchNormalization(n_in)
			conv_layer = ResidualConvLayer(**attributes)
			conv_layer.apply_batchnorm = conf.apply_batchnorm
			self.residual_conv_layers.append(conv_layer)

	@property
	def input_receptive_size(self):
		conf = self.conf
		return conf.residual_conv_kernel_size ** (len(conf.residual_conv_channels) + 1)

	@property
	def gpu_enabled(self):
		return self.conf.gpu_enabled
	

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