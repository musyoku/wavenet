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
		# Note: kernel_height is fixed to 1
		self.causal_conv_kernel_width = 2
		# [<- input   output ->]
		# audio input -> conv -> (128,) -> residual dilated conv stack
		# to stack more layers, [128, 128, 128, ...]
		self.causal_conv_channels = [128]

		self.residual_conv_dilation_no_bias = True
		self.residual_conv_projection_no_bias = True
		# Note: kernel_height is fixed to 1
		self.residual_conv_kernel_width = 2
		# [<- input   output ->]
		# causal conv output (128,) -> conv -> (32,) -> 1x1 conv -> (128,) -> conv -> (32,) -> 1x1 conv -> (128,) -> ...
		self.residual_conv_channels = [32, 32, 32, 32, 32, 32, 32, 32, 32]
		# [<- input   output ->]
		self.residual_conv_dilations = [1, 2, 4, 8, 2, 4, 8, 4, 8]

		self.softmax_conv_no_bias = False
		# Note: kernel_height is fixed to 1
		self.softmax_wscale = 0.01
		self.softmax_conv_kernel_width = 2
		# [<- input   output ->]
		# skip-connections -> ReLU -> conv -> (128,) -> ReLU -> conv -> (256,) -> softmax -> prediction
		self.softmax_conv_channels = [128, 256]

		self.gpu_enabled = True
		self.learning_rate = 0.001
		self.weight_decay = 0.00001
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
		if self.causal_conv_channels[-1] != self.softmax_conv_channels[0]:
			raise Exception("causal_conv_channels[-1] != softmax_conv_channels[0]")
		if self.audio_channels != self.softmax_conv_channels[-1]:
			raise Exception("audio_channels != softmax_conv_channels[-1]")
		if len(self.residual_conv_channels) != len(self.residual_conv_dilations):
			raise Exception("len(residual_conv_channels) != len(residual_conv_dilations)")
		for dilation in self.residual_conv_dilations:
			if bin(dilation).count("1") != 1:
				raise Exception("dilation must be 2 ** n")

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

class CausalPadding1d(function.Function):
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

class CausalSlice1d(function.Function):
	def __init__(self, cut=1):
		if cut < 1:
			raise Exception("CausalSlice1d: cut cannot be less than one.")
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
		output = x[:,:,:,self.cut:]
		return output,

	def backward(self, inputs, grad_outputs):
		xp = cuda.get_array_module(inputs[0])
		x, = inputs
		grad = xp.zeros(x.shape, dtype=xp.float32)
		grad[:,:,:,self.cut:] = grad_outputs[0]
		return grad,

class DilatedConvolution1D(L.Convolution2D):

	def __init__(self, in_channels, out_channels, ksize, kernel_width=2, dilation=1, stride=1, nobias=False, initialW=None):
		self.kernel_width = kernel_width
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.dilation = dilation
		super(DilatedConvolution1D, self).__init__(in_channels, out_channels, ksize=ksize, stride=stride, initialW=initialW, nobias=nobias)

	def padding_1d(self, x, pad):
		return CausalPadding1d(pad)(x)

	def slice_1d(self, x, cut):
		return CausalSlice1d(cut)(x)

	def __call__(self, x):
		batchsize = x.data.shape[0]
		input_x_width = x.data.shape[3]

		if self.dilation == 1:
			# perform normal convolution
			padded_x = self.padding_1d(x, self.kernel_width - 1)
			# print "padded_x:"
			# print padded_x.data
			out =  super(DilatedConvolution1D, self).__call__(padded_x)
			# print "out:"
			# print out.data
			return out

		# print "dilated-conv:"
		# print "	", self.in_channels
		# print "	", input_x_width
		# print "	", self.dilation

		# padding
		# # of elements in padded_x >= input_x_width + dilation * (kernel_width - 1) 
		# print "pad:"
		pad = self.dilation * (self.kernel_width - 1)
		# print "	", pad
		padded_x_width = input_x_width + pad
		# we need more padding to perform convlution
		if padded_x_width < self.kernel_width * self.dilation:
			pad += self.kernel_width * self.dilation - padded_x_width
			# print "	", pad
			padded_x_width = input_x_width + pad
		mod = padded_x_width % self.dilation
		if mod > 0:
			pad += self.dilation - mod
		# print "	", pad
		padded_x = self.padding_1d(x, pad)
		# print "padded_x:"
		# print padded_x.data

		# to skip (dilation - 1) elements
		padded_x = F.reshape(padded_x, (batchsize, self.in_channels, -1, self.dilation))
		# we can remove transpose operation when residual_conv_kernel_width is set to the kernel's height
		# padded_x = F.transpose(padded_x, (0, 1, 3, 2))
		# print "padded_x(reshaped):"
		# print padded_x.data

		# convolution
		out = super(DilatedConvolution1D, self).__call__(padded_x)
		# print "out:"
		# print out.data

		# reshape to the original shape
		out = F.reshape(out, (batchsize, self.out_channels, 1, -1))
		# print "out(reshaped):"
		# print out.data

		# remove padded elements
		cut = out.data.shape[3] - input_x_width
		# print "cut:"
		# print cut
		if cut > 0:
			out = self.slice_1d(out, cut)
		# print "out(cut):"
		# print out.data

		return out

class ResidualConvLayer(chainer.Chain):
	def __init__(self, **layers):
		super(ResidualConvLayer, self).__init__(**layers)
		self.dilation = 1

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	def __call__(self, x):
		# gated activation
		z = F.tanh(self.wf(x)) * F.sigmoid(self.wg(x))

		# 1x1 conv
		z = self.projection(z)

		# residual
		output = z + x
		return output, z

class WaveNet():

	def __init__(self, params):
		params.check()
		self.params = params
		self.create_network()
		self.setup_optimizers()

	def create_network(self):
		params = self.params

		# stack causal blocks
		self.causal_conv_layers = []
		nobias = params.causal_conv_no_bias
		kernel_width = params.causal_conv_kernel_width
		ksize = (1, kernel_width)

		channels = [(params.audio_channels, params.causal_conv_channels[0])]
		channels += zip(params.causal_conv_channels[:-1], params.causal_conv_channels[1:])

		for i, (n_in, n_out) in enumerate(channels):
			attributes = {}
			initial_w = np.random.normal(scale=math.sqrt(2.0 / (kernel_width * kernel_width * n_out)), size=(n_out, n_in, 1, kernel_width))

			layer = DilatedConvolution1D(n_in, n_out, ksize, 
				kernel_width=kernel_width, 
				dilation=1, 
				stride=1, nobias=nobias, initialW=initial_w)
			if params.gpu_enabled:
				layer.to_gpu()
			self.causal_conv_layers.append(layer)

		# stack residual blocks
		self.residual_conv_layers = []
		nobias_dilation = params.residual_conv_dilation_no_bias
		nobias_projection = params.residual_conv_projection_no_bias
		kernel_width = params.residual_conv_kernel_width
		n_in = params.causal_conv_channels[-1]

		for i in xrange(len(params.residual_conv_channels)):
			n_out = params.residual_conv_channels[i]
			dilation = params.residual_conv_dilations[i]
			# kernel
			if dilation == 1:
				ksize = (1, kernel_width)
				shape_w = (n_out, n_in, 1, kernel_width)
			else:
				# set kernel_width to kernel's height to remove transpose operation in DilatedConvolution1D
				ksize = (kernel_width, 1)
				shape_w = (n_out, n_in, kernel_width, 1)

			attributes = {}

			# weight for filter
			initial_w = np.random.normal(scale=math.sqrt(2.0 / (kernel_width * kernel_width * n_out)), size=shape_w)
			# initial_w = np.ones(shape_w).astype(np.float32)

			# filter
			dilated_conv_layer = DilatedConvolution1D(n_in, n_out, ksize, 
				kernel_width=kernel_width,
				dilation=dilation,
				stride=1, nobias=nobias_dilation, initialW=initial_w)
			dilated_conv_layer.kernel_width = kernel_width
			dilated_conv_layer.dilation = dilation
			attributes["wf"] = dilated_conv_layer 

			# weight for gate
			initial_w = np.random.normal(scale=math.sqrt(2.0 / (kernel_width * kernel_width * n_out)), size=shape_w)
			# initial_w = np.ones(shape_w).astype(np.float32)

			# gate
			dilated_conv_layer = DilatedConvolution1D(n_in, n_out, ksize, 
				kernel_width=kernel_width, 
				dilation=dilation,
				stride=1, nobias=nobias_dilation, initialW=initial_w)
			dilated_conv_layer.dilation = dilation
			dilated_conv_layer.kernel_width = kernel_width
			attributes["wg"] = dilated_conv_layer

			# projection
			attributes["projection"] = L.Convolution2D(n_out, n_in, 1, stride=1, nobias=nobias_projection)

			# residual conv block
			residual_layer = ResidualConvLayer(**attributes)
			if params.gpu_enabled:
				residual_layer.to_gpu()
			self.residual_conv_layers.append(residual_layer)

		# softmax block
		self.softmax_conv_layers = []
		nobias = params.softmax_conv_no_bias

		channels = [(params.causal_conv_channels[-1], params.softmax_conv_channels[0])]
		channels += zip(params.softmax_conv_channels[:-1], params.softmax_conv_channels[1:])

		for i, (n_in, n_out) in enumerate(channels):
			initial_w = np.random.normal(scale=math.sqrt(2.0 / (kernel_width * kernel_width * n_out)), size=(n_out, n_in, 1, 1))

			conv_layer = L.Convolution2D(n_in, n_out, ksize=1, stride=1, nobias=nobias, initialW=initial_w)
			if params.gpu_enabled:
				conv_layer.to_gpu()
			self.softmax_conv_layers.append(conv_layer)

	def setup_optimizers(self):
		params = self.params
		
		self.causal_conv_optimizers = []
		for layer in self.causal_conv_layers:
			opt = optimizers.Adam(alpha=params.learning_rate, beta1=params.gradient_momentum)
			opt.setup(layer)
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
			opt.add_hook(GradientClipping(params.gradient_clipping))
			self.causal_conv_optimizers.append(opt)
		
		self.residual_conv_optimizers = []
		for layer in self.residual_conv_layers:
			opt = optimizers.Adam(alpha=params.learning_rate, beta1=params.gradient_momentum)
			opt.setup(layer)
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
			opt.add_hook(GradientClipping(params.gradient_clipping))
			self.residual_conv_optimizers.append(opt)
		
		self.softmax_conv_optimizers = []
		for layer in self.softmax_conv_layers:
			opt = optimizers.Adam(alpha=params.learning_rate, beta1=params.gradient_momentum)
			opt.setup(layer)
			opt.add_hook(optimizer.WeightDecay(params.weight_decay))
			opt.add_hook(GradientClipping(params.gradient_clipping))
			self.softmax_conv_optimizers.append(opt)

	def zero_grads(self):
		for opt in self.causal_conv_optimizers:
			opt.zero_grads()

		for opt in self.residual_conv_optimizers:
			opt.zero_grads()
			
		for opt in self.softmax_conv_optimizers:
			opt.zero_grads()

	def update(self):
		for opt in self.causal_conv_optimizers:
			opt.update()

		for opt in self.residual_conv_optimizers:
			opt.update()
			
		for opt in self.softmax_conv_optimizers:
			opt.update()

	@property
	def gpu_enabled(self):
		return self.params.gpu_enabled

	def forward_one_step(self, padded_x_batch_data, softmax=True):
		x_batch = Variable(padded_x_batch_data)
		if self.gpu_enabled:
			x_batch.to_gpu()
		causal_output = self.forward_causal_block(x_batch)
		residual_output, sum_skip_connections = self.forward_residual_block(causal_output)
		softmax_output = self.forward_softmax_block(residual_output, softmax=softmax)
		return softmax_output

	def forward_causal_block(self, x_batch):
		for layer in self.causal_conv_layers:
			output = layer(x_batch)
		return output

	def forward_residual_block(self, x_batch):
		# print "x_batch:"
		# print x_batch.data
		sum_skip_connections = 0
		for layer in self.residual_conv_layers:
			output, z = layer(x_batch)
			sum_skip_connections += z
			# print "output:"
			# print "	", output.data
			# print "z:"
			# print "	", z.data
		return output, sum_skip_connections

	def forward_softmax_block(self, x_batch, softmax=True):
		batchsize = x_batch.data.shape[0]
		for layer in self.softmax_conv_layers:
			output = layer(x_batch)
		if softmax:
			output = F.softmax(output)
		return output

	# padded_input_batch_data: 			(batchsize, channels, 1, time_step)
	# target_signal_batch_data.ndim:	(batchsize, time_step)
	def loss(self, padded_input_batch_data, target_signal_batch_data):
		batchsize = padded_input_batch_data.shape[0]
		width = target_signal_batch_data.shape[1]
		raw_output = self.forward_one_step(padded_input_batch_data, softmax=True)
		print "prob:"
		print raw_output.data
		raw_output = self.forward_one_step(padded_input_batch_data, softmax=False)
		print "raw:"
		print raw_output.data.shape
		print target_signal_batch_data
		cut = padded_input_batch_data.shape[3] - width
		if cut > 0:
			raw_output = CausalSlice1d(cut)(raw_output)
		print raw_output.data

		# (batchsize * time_step,) <- (batchsize, time_step)
		target_signal_batch_data = target_signal_batch_data.reshape((-1,))
		print target_signal_batch_data

		# (batchsize, channels, 1, time_step)
		raw_output = F.transpose(raw_output, (0, 3, 2, 1))
		print raw_output.data
		raw_output = F.reshape(raw_output, (batchsize * width, -1))
		print raw_output.data

		target_id_batch = Variable(target_signal_batch_data)
		if self.gpu_enabled:
			target_id_batch.to_gpu()

		loss = F.sum(F.softmax_cross_entropy(raw_output, target_id_batch))
		return loss

	def backprop(self, loss):
		self.zero_grads()
		loss.backward()
		self.update()

	def save(self, dir="./"):
		pass

	def load(self, dir="./"):
		pass