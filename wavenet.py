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

		self.conv_kernel_sizes= [2, 2, 2]
		self.conv_out_channels= [256, 256, 256]

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
		if len(self.conv_kernel_sizes) != len(self.conv_out_channels):
			raise Exception("len(self.conv_kernel_sizes) != len(self.conv_out_channels)")

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
		# 1x1 conv
		z = self.projection(z)
		# residual
		output = z + x
		return output, z

class WaveNet():

	def __init__(self, conf):
		conf.check()
		self.conf = conf

		# stack residual blocks
		self.conv_layers = []
		channels = [(conf.audio_channels, conf.conv_out_channels[0])]
		channels += zip(conf.conv_out_channels[:-1], conf.conv_out_channels[1:])
		for i, (n_in, n_out) in enumerate(channels):
			ksize = conf.conv_kernel_sizes[i]
			attributes = {}
			attributes["wf"] = L.Convolution2D(n_in, n_out, ksize, stride=ksize)
			attributes["wg"] = L.Convolution2D(n_in, n_out, ksize, stride=ksize)
			attributes["projection"] = L.Convolution2D(n_out, n_in, 1, stride=1)
			attributes["batchnorm"] = L.BatchNormalization(n_in)
			conv_layer = ResidualConvLayer(**attributes)
			conv_layer.apply_batchnorm = conf.apply_batchnorm
			self.conv_layers.append(conv_layer)

	def save(self, dir="./"):
		pass

	def load(self, dir="./"):
		pass