# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time
from chainer import cuda, Variable, optimizers, serializers, function, optimizer
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L

class Eve(optimizer.GradientMethod):
	def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, beta3=0.999, eps=1e-8, lower_threshold=0.1, upper_threshold=10):
		self.alpha = alpha
		self.beta1 = beta1
		self.beta2 = beta2
		self.beta3 = beta3
		self.eps = eps
		self.lower_threshold = lower_threshold
		self.upper_threshold = upper_threshold

	def init_state(self, param, state):
		xp = cuda.get_array_module(param.data)
		with cuda.get_device(param.data):
			state['m'] = xp.zeros_like(param.data)
			state['v'] = xp.zeros_like(param.data)
			state['d'] = xp.ones(1, dtype=param.data.dtype)
			state['f'] = xp.zeros(1, dtype=param.data.dtype)

	def _update_d_and_f(self, state):
		d, f = state['d'], state['f']
		if self.t > 1:
			old_f = float(cuda.to_cpu(state['f']))
			if self.loss > old_f:
				delta = self.lower_threshold + 1.
				Delta = self.upper_threshold + 1.
			else:
				delta = 1. / (self.upper_threshold + 1.)
				Delta = 1. / (self.lower_threshold + 1.)
			c = min(max(delta, self.loss / (old_f + 1e-12)), Delta)
			new_f = c * old_f
			r = abs(new_f - old_f) / (min(new_f, old_f) + 1e-12)
			d += (1 - self.beta3) * (r - d)
			f[:] = new_f
		else:
			f[:] = self.loss

	def update_one_cpu(self, param, state):
		m, v, d = state['m'], state['v'], state['d']
		grad = param.grad

		self._update_d_and_f(state)
		m += (1. - self.beta1) * (grad - m)
		v += (1. - self.beta2) * (grad * grad - v)
		param.data -= self.lr * m / (d * np.sqrt(v) + self.eps)

	def update_one_gpu(self, param, state):
		self._update_d_and_f(state)
		cuda.elementwise(
			'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps, T d',
			'T param, T m, T v',
			'''m += one_minus_beta1 * (grad - m);
			   v += one_minus_beta2 * (grad * grad - v);
			   param -= lr * m / (d * sqrt(v) + eps);''',
			'eve')(param.grad, self.lr, 1 - self.beta1, 1 - self.beta2,
				   self.eps, float(state['d']), param.data, state['m'],
				   state['v'])

	@property
	def lr(self):
		fix1 = 1. - self.beta1 ** self.t
		fix2 = 1. - self.beta2 ** self.t
		return self.alpha * math.sqrt(fix2) / fix1

	def update(self, lossfun=None, *args, **kwds):
		# Overwrites GradientMethod.update in order to get loss values
		if lossfun is None:
			raise RuntimeError('Eve.update requires lossfun to be specified')
		loss_var = lossfun(*args, **kwds)
		self.loss = float(loss_var.data)
		super(Eve, self).update(lossfun=lambda: loss_var)

def get_optimizer(name, lr, momentum=0.9):
	if name.lower() == "adam":
		return chainer.optimizers.Adam(alpha=lr, beta1=momentum)
	if name.lower() == "eve":
		return Eve(alpha=lr, beta1=momentum)
	if name.lower() == "adagrad":
		return chainer.optimizers.AdaGrad(lr=lr)
	if name.lower() == "adadelta":
		return chainer.optimizers.AdaDelta(rho=momentum)
	if name.lower() == "nesterov" or name.lower() == "nesterovag":
		return chainer.optimizers.NesterovAG(lr=lr, momentum=momentum)
	if name.lower() == "rmsprop":
		return chainer.optimizers.RMSprop(lr=lr, alpha=momentum)
	if name.lower() == "momentumsgd":
		return chainer.optimizers.MomentumSGD(lr=lr, mommentum=mommentum)
	if name.lower() == "sgd":
		return chainer.optimizers.SGD(lr=lr)
	raise Exception()

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

		self.optimizer = "adam"
		self.weight_decay = 0
		self.momentum = 0.9
		self.gradient_clipping = 1.0

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

# [1, 2, 3, 4] -> [0, 0, 0, 1, 2, 3, 4]
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

# [1, 2, 3, 4] -> [3, 4]
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

	def __init__(self, in_channels, out_channels, ksize, filter_width=2, dilation=1, stride=1, nobias=False):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.filter_width = filter_width
		self.dilation = dilation
		super(DilatedConvolution1D, self).__init__(in_channels, out_channels, ksize=ksize, stride=stride, nobias=nobias)

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

class ResidualConvLayer():
	@property
	def xp(self):
		return np if self._cpu else cuda.cupy

	# for faster generation
	def _forward(self, x):
		z = F.tanh(self.wf._forward(x)) * F.sigmoid(self.wg._forward(x))
		projection_block = self.projection_block(z)
		projection_softmax = self.projection_softmax(z)
		output = projection_block.data[0, :, 0, 0] + x[0, :, 0, -1]
		output = output.reshape((1, -1, 1, 1))
		return output, projection_softmax.data

	def __call__(self, x):
		# gated activation
		z = F.tanh(self.wf(x)) * F.sigmoid(self.wg(x))

		# 1x1 conv
		projection_block = self.projection_block(z)
		projection_softmax = self.projection_softmax(z)

		# residual
		output = projection_block + x
		return output, projection_softmax

class WaveNet():
	def __init__(self, params):
		params.check()
		self.params = params
		self.chain = chainer.Chain()
		self.create_network()
		self.setup_optimizer()
		self._gpu = False

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
			layer = DilatedConvolution1D(n_in, n_out, ksize, 
					filter_width=filter_width,
					dilation=1,
					nobias=nobias)
			self.causal_conv_layers.append(layer)

		# stack residual blocks
		self.residual_blocks = []
		nobias_dilation = params.residual_conv_dilation_no_bias
		nobias_projection = params.residual_conv_projection_no_bias
		filter_width = params.residual_conv_filter_width
		n_in = params.causal_conv_channels[-1]
		n_softmax_in = params.softmax_conv_channels[0]

		residual_conv_dilations = []
		dilation = 1
		for _ in params.residual_conv_channels:
			residual_conv_dilations.append(dilation)
			dilation *= filter_width

		for stack in xrange(params.residual_num_blocks):
			residual_conv_layers = []
			for layer_index in xrange(len(params.residual_conv_channels)):
				residual_layer = ResidualConvLayer()
				n_out = params.residual_conv_channels[layer_index]
				# filter
				if layer_index == 0:
					ksize = (1, filter_width)
					shape_w = (n_out, n_in, 1, filter_width)
				else:
					# set filter_width to filter's height to remove transpose operation in DilatedConvolution1D
					ksize = (filter_width, 1)
					shape_w = (n_out, n_in, filter_width, 1)

				# filter
				residual_layer.wf = DilatedConvolution1D(n_in, n_out, ksize, 
					filter_width=filter_width,
					dilation=filter_width ** layer_index,
					nobias=nobias_dilation)

				# gate
				residual_layer.wg = DilatedConvolution1D(n_in, n_out, ksize, 
					filter_width=filter_width,
					dilation=filter_width ** layer_index,
					nobias=nobias_dilation)

				# projection
				residual_layer.projection_block = L.Convolution2D(n_out, n_in, 1, stride=1, pad=0, nobias=nobias_projection)
				residual_layer.projection_softmax = L.Convolution2D(n_out, n_softmax_in, 1, stride=1, pad=0, nobias=nobias_projection)

				# residual conv block
				residual_conv_layers.append(residual_layer)
			self.residual_blocks.append(residual_conv_layers)

		# softmax block
		self.softmax_conv_layers = []
		nobias = params.softmax_conv_no_bias

		# channels = [(params.causal_conv_channels[-1], params.softmax_conv_channels[0])]
		channels = zip(params.softmax_conv_channels[:-1], params.softmax_conv_channels[1:])

		for i, (n_in, n_out) in enumerate(channels):
			# initial_w = np.random.normal(scale=math.sqrt(2.0 / n_out), size=(n_out, n_in, 1, 1))
			self.softmax_conv_layers.append(L.Convolution2D(n_in, n_out, ksize=1, stride=1, pad=0, nobias=nobias))

	def setup_optimizer(self):
		params = self.params
		
		# add all links
		for i, link in enumerate(self.causal_conv_layers):
			self.chain.add_link("causal_{}".format(i), link)
		
		for j, block in enumerate(self.residual_blocks):
			for i, layer in enumerate(block):
				self.chain.add_link("residual_{}_block_{}_wf".format(j, i), layer.wf)
				self.chain.add_link("residual_{}_block_{}_wg".format(j, i), layer.wg)
				self.chain.add_link("residual_{}_block_{}_projection_block".format(j, i), layer.projection_block)
				self.chain.add_link("residual_{}_block_{}_projection_softmax".format(j, i), layer.projection_softmax)
		
		for i, link in enumerate(self.softmax_conv_layers):
			self.chain.add_link("softmax_{}".format(i), link)

		# setup optimizer
		opt = get_optimizer(params.optimizer, 0.0001, params.momentum)
		opt.setup(self.chain)
		if params.weight_decay > 0:
			opt.add_hook(chainer.optimizer.WeightDecay(params.weight_decay))
		if params.gradient_clipping > 0:
			opt.add_hook(GradientClipping(params.gradient_clipping))
		self.optimizer = opt

	def update_laerning_rate(self, lr):
		if isinstance(self.optimizer, optimizers.Adam):
			self.optimizer.alpha = lr
			return
		if isinstance(self.optimizer, Eve):
			self.optimizer.alpha = lr
			return
		if isinstance(self.optimizer, optimizers.AdaDelta):
			# AdaDelta has no learning rate
			return
		self.optimizer.lr = lr

	def update_momentum(self, momentum):
		if isinstance(self.optimizer, optimizers.Adam):
			self.optimizer.beta1 = momentum
			return
		if isinstance(self.optimizer, Eve):
			self.optimizer.beta1 = momentum
			return
		if isinstance(self.optimizer, optimizers.AdaDelta):
			self.optimizer.rho = momentum
			return
		if isinstance(self.optimizer, optimizers.NesterovAG):
			self.optimizer.momentum = momentum
			return
		if isinstance(self.optimizer, optimizers.RMSprop):
			self.optimizer.alpha = momentum
			return
		if isinstance(self.optimizer, optimizers.MomentumSGD):
			self.optimizer.mommentum = momentum
			return

	def backprop(self, loss):
		if isinstance(loss, Variable):
			self.optimizer.update(lossfun=lambda: loss)
		else:
			self.optimizer.update(lossfun=loss)

	def to_gpu(self):
		self.chain.to_gpu()
		self._gpu = True

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

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

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		if isinstance(x, Variable):
			return x.data.shape[0]
		return x.shape[0]

	def forward_one_step(self, x_batch, apply_softmax=True, as_numpy=False):
		x_batch = self.to_variable(x_batch)
		causal_output = self.forward_causal_block(x_batch)
		residual_output, sum_skip_connections = self.forward_residual_block(causal_output)
		softmax_output = self.forward_softmax_block(sum_skip_connections, apply_softmax=apply_softmax)
		if as_numpy:
			return self.to_numpy(softmax_output)
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

	def forward_softmax_block(self, x_batch, apply_softmax=True):
		input_batch = self.to_variable(x_batch)
		batchsize = self.get_batchsize(x_batch)
		for layer in self.softmax_conv_layers:
			input_batch = F.relu(input_batch)
			output = layer(input_batch)
			input_batch = output
		if apply_softmax:
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

		loss = F.softmax_cross_entropy(raw_network_output, target_signal)
		return loss

	def save(self, model_dir="./"):
		try:
			os.mkdir(model_dir)
		except:
			pass
		serializers.save_hdf5(model_dir + "/wavenet.model", self.chain)
		serializers.save_hdf5(model_dir + "/wavenet.opt", self.optimizer)

	def load(self, model_dir="./"):
		filename = model_dir + "/wavenet.model"
		if os.path.isfile(filename):
			print "loading", filename, "..."
			serializers.load_hdf5(filename, self.chain)
		else:
			pass
		filename = model_dir + "/wavenet.opt"
		if os.path.isfile(filename):
			print "loading", filename, "..."
			serializers.load_hdf5(filename, self.optimizer)
		else:
			pass
