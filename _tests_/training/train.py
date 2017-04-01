from scipy.io import wavfile
import numpy as np
import os, sys
from chainer import cuda
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from args import args
from model import params, wavenet
import data

updates = np.zeros(1000, dtype=int)


def create_batch(signal, batch_size, receptive_width, target_width):
	indecis = np.random.randint(0, signal.size - target_width - receptive_width, size=batch_size)
	input_batch = np.empty((batch_size, receptive_width + target_width), dtype=np.int32)
	target_batch = np.empty((batch_size, target_width), dtype=np.int32)
	for n in xrange(batch_size):
		start = indecis[n]
		input_batch[n] = signal[start:start + receptive_width + target_width]
		target_batch[n] = signal[start + receptive_width + 1:start + receptive_width + target_width + 1]
		updates[start + receptive_width + 1 - target_width:start + receptive_width + 1] += 1
	return input_batch, target_batch

def train_audio():
	# compute required input width
	num_layers = len(params.residual_conv_channels)
	receptive_width_per_unit = params.residual_conv_filter_width ** num_layers
	receptive_width = (receptive_width_per_unit - 1) * params.residual_num_blocks + 1
	# padding for causal conv block
	causal_padding = len(params.causal_conv_channels)
	

	# quantized_signal = np.mod(np.arange(1, 100), 6)
	quantized_signal = np.repeat(np.arange(0, 10), 100, axis=0)
	# quantized_signal = np.random.randint(0, params.quantization_steps, 1000)
	original_signal_width = quantized_signal.size
	quantized_signal = np.insert(quantized_signal, 0, np.full((receptive_width + causal_padding,), 0, dtype=np.int32), axis=0)

	target_width = original_signal_width // 20
	batch_size = 2

	for epoch in xrange(100):
		sum_loss = 0
		for step in xrange(500):
			input_batch, target_batch = create_batch(quantized_signal, batch_size, receptive_width + causal_padding, target_width)

			padded_onehot_batch = data.onehot_pixel_image(input_batch, quantization_steps=params.quantization_steps)

			# convert to 1xW image whose #channels is equal to the quantization steps of audio
			# input_batch.shape = (BATCHSIZE, CHANNELS(=quantization_steps), HEIGHT(=1), WIDTH(=input_width))
			input_batch = data.onehot_pixel_image(input_batch, quantization_steps=params.quantization_steps)

			# training
			## causal block
			output = wavenet.forward_causal_block(input_batch)
			## remove causal padding
			# output = wavenet.slice_1d(output, len(params.causal_conv_channels))
			## residual dilated conv block
			output, sum_skip_connections = wavenet.forward_residual_block(output)
			## remove unnecessary elements
			sum_skip_connections = wavenet.slice_1d(sum_skip_connections, sum_skip_connections.data.shape[3] - target_width)
			## softmax block
			## Note: do not apply F.softmax
			output = wavenet.forward_softmax_block(sum_skip_connections, apply_softmax=False)

			## compute cross entroy
			loss = wavenet.cross_entropy(output, target_batch)
			## update weights
			wavenet.backprop(loss)
			sum_loss += float(loss.data)
		print epoch, sum_loss
		wavenet.save(args.model_dir)

def main():
	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	train_audio()

if __name__ == '__main__':
	main()
