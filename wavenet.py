# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L

class Params():
	def __init__(self, dict=None):
		self.quantization_steps = 256
		self.sampling_rate = 8000

		# entire architecture
		# causal conv stack -> residual dilated conv stack -> skip-connections conv -> softmax

		self.causal_conv_no_bias = True
		# Note: kernel_height is fixed to 1
		self.causal_conv_filter_width = 2
		# [<- input   output ->]
		# audio input -> conv -> (128,) -> residual dilated conv stack
		# to add more layers, [128, 128, 128, ...]
		self.causal_conv_channels = [128]

		self.residual_conv_dilation_no_bias = True
		self.residual_conv_projection_no_bias = True
		# Note: kernel_height is fixed to 1
		self.residual_conv_filter_width = 2
		# [<- input   output ->]
		# causal conv output (128,) -> conv -> (32,) -> 1x1 conv -> (128,) -> conv -> (32,) -> 1x1 conv -> (128,) -> ...
		# dilation will be determined automatically by the length of residual_conv_channels
		# e.g.             dilation = [1,  2,  4,  8,  16, 32, 64,128,256]
		self.residual_conv_channels = [32, 32, 32, 32, 32, 32, 32, 32, 32]
		# e.g.
		# residual_conv_channels = [16, 16] and residual_num_blocks = 3
		# 
		#                                #1                      #2                      #3
		#                        dilation 1, 2, ...      dilation 1, 2, ...      dilation 1, 2, ...
		# causal conv output -> {conv 16 -> conv 16} -> {conv 16 -> conv 16} -> {conv 16 -> conv 16} -> output (it will be ignored)
		#                           |          |            |          |            |          |
		#                           +----------+------------+----------+------------+----------+-> skip connection -> softmax
		# 

		# deeper network and wider receptive field
		self.residual_num_blocks = 2

		self.softmax_conv_no_bias = False
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
			if hasattr(self, attr):
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
		if self.quantization_steps != self.softmax_conv_channels[-1]:
			raise Exception("quantization_steps != softmax_conv_channels[-1]")

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

	def __init__(self, in_channels, out_channels, ksize, filter_width=2, layer_index=0, stride=1, nobias=False, initialW=None):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.filter_width = filter_width
		self.dilation = filter_width ** layer_index
		super(DilatedConvolution1D, self).__init__(in_channels, out_channels, ksize=ksize, stride=stride, initialW=initialW, nobias=nobias)

	# [1, 2, 3, 4] -> [0, 0, 0, 1, 2, 3, 4]
	def padding_1d(self, x, pad):
		return CausalPadding1d(pad)(x)

	# [1, 2, 3, 4] -> [3, 4]
	def slice_1d(self, x, cut):
		return CausalSlice1d(cut)(x)

	# for faster generation
	def _forward(self, x_batch_data):
		xp = cuda.get_array_module(x_batch_data)
		if self.dilation == 1:
			x = xp.empty((1, self.in_channels, 1, self.filter_width), dtype=xp.float32)
			for n in xrange(self.filter_width):
				x[0, :, 0, -n - 1] = x_batch_data[0, :, 0, -self.dilation * n - 1]
		else:
			x = xp.empty((1, self.in_channels, self.filter_width, 1), dtype=xp.float32)
			for n in xrange(self.filter_width):
				x[0, :, -n - 1, 0] = x_batch_data[0, :, 0, -self.dilation * n - 1]

		return super(DilatedConvolution1D, self).__call__(Variable(x)).data

	def __call__(self, x):
		batchsize = x.data.shape[0]
		input_x_width = x.data.shape[3]

		if self.dilation == 1:
			# perform normal convolution
			padded_x = self.padding_1d(x, self.filter_width - 1)
			return super(DilatedConvolution1D, self).__call__(padded_x)

		# padding
		pad = 0
		padded_x_width = input_x_width

		## check if we can reshape
		mod = padded_x_width % self.dilation
		if mod > 0:
			pad += self.dilation - mod
			padded_x_width = input_x_width + pad

		## check if height < filter width
		height = padded_x_width / self.dilation
		if height < self.filter_width:
			pad += (self.filter_width - height) * self.dilation
			padded_x_width = input_x_width + pad

		if pad > 0:
			padded_x = self.padding_1d(x, pad)
		else:
			padded_x = x

		# to skip (dilation - 1) elements
		padded_x = F.reshape(padded_x, (batchsize, self.in_channels, -1, self.dilation))
		# we can remove transpose operation when residual_conv_filter_width is set to the kernel's height
		# padded_x = F.transpose(padded_x, (0, 1, 3, 2))

		# convolution
		out = super(DilatedConvolution1D, self).__call__(padded_x)

		# reshape to the original shape
		out = F.reshape(out, (batchsize, self.out_channels, 1, -1))

		# remove padded elements / add missing elements
		cut = out.data.shape[3] - input_x_width
		if cut > 0:
			out = self.slice_1d(out, cut)
		elif cut < 0:
			out = self.padding_1d(out, -cut)

		return out

class ResidualConvLayer(chainer.Chain):
	def __init__(self, **layers):
		super(ResidualConvLayer, self).__init__(**layers)

	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	# for faster generation
	def _forward(self, x):
		z = F.tanh(self.wf._forward(x)) * F.sigmoid(self.wg._forward(x))
		z = self.projection(z)
		output = z.data[0, :, 0, 0] + x[0, :, 0, -1]
		output = output.reshape((1, -1, 1, 1))
		return output, z.data

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
		filter_width = params.causal_conv_filter_width
		ksize = (1, filter_width)

		channels = [(params.quantization_steps, params.causal_conv_channels[0])]
		channels += zip(params.causal_conv_channels[:-1], params.causal_conv_channels[1:])

		for layer_index, (n_in, n_out) in enumerate(channels):
			attributes = {}
			std = math.sqrt(2.0 / (filter_width * filter_width * n_in))
			initial_w = np.random.normal(scale=std, size=(n_out, n_in, 1, filter_width))

			layer = DilatedConvolution1D(n_in, n_out, ksize, 
				filter_width=filter_width, 
				layer_index=layer_index, 
				stride=1, nobias=nobias, initialW=initial_w)
			if params.gpu_enabled:
				layer.to_gpu()
			self.causal_conv_layers.append(layer)

		# stack residual blocks
		self.residual_blocks = []
		nobias_dilation = params.residual_conv_dilation_no_bias
		nobias_projection = params.residual_conv_projection_no_bias
		filter_width = params.residual_conv_filter_width
		n_in = params.causal_conv_channels[-1]

		residual_conv_dilations = []
		dilation = 1
		for _ in params.residual_conv_channels:
			residual_conv_dilations.append(dilation)
			dilation *= filter_width

		for stack in xrange(params.residual_num_blocks):
			residual_conv_layers = []
			for layer_index in xrange(len(params.residual_conv_channels)):
				n_out = params.residual_conv_channels[layer_index]
				# filter
				if layer_index == 0:
					ksize = (1, filter_width)
					shape_w = (n_out, n_in, 1, filter_width)
				else:
					# set filter_width to filter's height to remove transpose operation in DilatedConvolution1D
					ksize = (filter_width, 1)
					shape_w = (n_out, n_in, filter_width, 1)

				attributes = {}

				# weight for filter
				std = math.sqrt(2.0 / (filter_width * filter_width * n_in))
				initial_w = np.random.normal(scale=std, size=shape_w)

				# filter
				dilated_conv_layer = DilatedConvolution1D(n_in, n_out, ksize, 
					filter_width=filter_width,
					layer_index=layer_index,
					stride=1, nobias=nobias_dilation, initialW=initial_w)
				attributes["wf"] = dilated_conv_layer 

				# weight for gate
				std = math.sqrt(2.0 / (filter_width * filter_width * n_in))
				initial_w = np.random.normal(scale=std, size=shape_w)

				# gate
				dilated_conv_layer = DilatedConvolution1D(n_in, n_out, ksize, 
					filter_width=filter_width, 
					layer_index=layer_index,
					stride=1, nobias=nobias_dilation, initialW=initial_w)
				attributes["wg"] = dilated_conv_layer

				# projection
				attributes["projection"] = L.Convolution2D(n_out, n_in, 1, stride=1, nobias=nobias_projection)

				# residual conv block
				residual_layer = ResidualConvLayer(**attributes)
				if params.gpu_enabled:
					residual_layer.to_gpu()
				residual_conv_layers.append(residual_layer)
			self.residual_blocks.append(residual_conv_layers)

		# softmax block
		self.softmax_conv_layers = []
		nobias = params.softmax_conv_no_bias

		channels = [(params.causal_conv_channels[-1], params.softmax_conv_channels[0])]
		channels += zip(params.softmax_conv_channels[:-1], params.softmax_conv_channels[1:])

		for i, (n_in, n_out) in enumerate(channels):
			initial_w = np.random.normal(scale=math.sqrt(2.0 / n_out), size=(n_out, n_in, 1, 1))

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
		for block in self.residual_blocks:
			for layer in block:
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

	def update_laerning_rate(self, lr):
		for opt in self.causal_conv_optimizers:
			opt.alpha = lr

		for opt in self.residual_conv_optimizers:
			opt.alpha = lr
			
		for opt in self.softmax_conv_optimizers:
			opt.alpha = lr

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

	def slice_1d(self, x, cut=0):
		return CausalSlice1d(cut)(x)

	def padding_1d(self, x, pad=0):
		return CausalPadding1d(pad)(x)

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def get_batchsize(self, x):
		if isinstance(x, Variable):
			return x.data.shape[0]
		return x.shape[0]

	def forward_one_step(self, x_batch, softmax=True, return_numpy=False):
		x_batch = self.to_variable(x_batch)
		causal_output = self.forward_causal_block(x_batch)
		residual_output, sum_skip_connections = self.forward_residual_block(causal_output)
		softmax_output = self.forward_softmax_block(sum_skip_connections, softmax=softmax)
		if return_numpy:
			if self.gpu_enabled:
				softmax_output.to_cpu()
			return softmax_output.data
		return softmax_output

	def forward_causal_block(self, x_batch):
		input_batch = self.to_variable(x_batch)
		for layer in self.causal_conv_layers:
			output = layer(input_batch)
			input_batch = output
		return output

	def forward_residual_block(self, x_batch):
		params = self.params
		sum_skip_connections = 0
		input_batch = self.to_variable(x_batch)
		for i, block in enumerate(self.residual_blocks):
			for layer in block:
				output, z = layer(input_batch)
				sum_skip_connections += z
				input_batch = output

		return output, sum_skip_connections

	def forward_softmax_block(self, x_batch, softmax=True):
		input_batch = self.to_variable(x_batch)
		batchsize = self.get_batchsize(x_batch)
		for layer in self.softmax_conv_layers:
			input_batch = F.elu(input_batch)
			output = layer(input_batch)
			input_batch = output
		if softmax:
			output = F.softmax(output)
		return output

	# raw_network_output.ndim:	(batchsize, channels, 1, time_step)
	# target_signal_data.ndim:	(batchsize, time_step)
	def cross_entropy(self, raw_network_output, target_signal_data):
		if isinstance(target_signal_data, Variable):
			raise Exception("target_signal_data cannot be Variable")

		raw_network_output = self.to_variable(raw_network_output)
		target_width = target_signal_data.shape[1]
		batchsize = raw_network_output.data.shape[0]

		if raw_network_output.data.shape[3] != target_width:
			raise Exception("raw_network_output.width != target.width")

		# (batchsize * time_step,) <- (batchsize, time_step)
		target_signal_data = target_signal_data.reshape((-1,))
		target_signal = self.to_variable(target_signal_data)

		# (batchsize * time_step, channels) <- (batchsize, channels, 1, time_step)
		raw_network_output = F.transpose(raw_network_output, (0, 3, 2, 1))
		raw_network_output = F.reshape(raw_network_output, (batchsize * target_width, -1))

		loss = F.sum(F.softmax_cross_entropy(raw_network_output, target_signal))
		return loss

	def backprop(self, loss):
		self.zero_grads()
		loss.backward()
		self.update()

	def save(self, dir="./"):
		try:
			os.mkdir(dir)
		except:
			pass
		for i, layer in enumerate(self.causal_conv_layers):
			filename = dir + "/causal_conv_layer_{}.hdf5".format(i)
			serializers.save_hdf5(filename, layer)

		for i, block in enumerate(self.residual_blocks):
			for j, layer in enumerate(block):
				filename = dir + "/residual_{}_conv_layer_{}.hdf5".format(i, j)
				serializers.save_hdf5(filename, layer)

		for i, layer in enumerate(self.softmax_conv_layers):
			filename = dir + "/softmax_conv_layer_{}.hdf5".format(i)
			serializers.save_hdf5(filename, layer)
			

	def load(self, dir="./"):
		def load_hdf5(filename, layer):
			if os.path.isfile(filename):
				print "loading", filename
				serializers.load_hdf5(filename, layer)
			
		for i, layer in enumerate(self.causal_conv_layers):
			filename = dir + "/causal_conv_layer_{}.hdf5".format(i)
			load_hdf5(filename, layer)
			
		for i, block in enumerate(self.residual_blocks):
			for j, layer in enumerate(block):
				filename = dir + "/residual_{}_conv_layer_{}.hdf5".format(i, j)
				load_hdf5(filename, layer)
			
		for i, layer in enumerate(self.softmax_conv_layers):
			filename = dir + "/softmax_conv_layer_{}.hdf5".format(i)
			load_hdf5(filename, layer)

