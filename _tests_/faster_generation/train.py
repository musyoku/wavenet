from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from args import args
from model import params, wavenet
import data

def create_batch(signal, batch_size, input_width, target_width):
	indecis = np.random.randint(0, signal.size - input_width - 1, size=batch_size)
	input_batch = np.empty((batch_size, input_width), dtype=np.int32)
	target_batch = np.empty((batch_size, target_width), dtype=np.int32)
	for n in xrange(batch_size):
		start = indecis[n]
		input_batch[n] = signal[start:start + input_width]
		target_batch[n] = signal[start + input_width + 1 - target_width:start + input_width + 1]
	return input_batch, target_batch

def train_audio():

	# compute receptive field width
	learnable_steps = 1
	batch_size = 1
	num_layers = len(params.residual_conv_channels)
	receptive_steps_per_unit = params.residual_conv_filter_width ** num_layers
	receptive_steps = (receptive_steps_per_unit - 1) * params.residual_num_blocks + 1
	target_width = learnable_steps
	input_width = receptive_steps
	# to compute all learnable targets
	input_width += learnable_steps - 1
	## padding for causal conv block
	input_width += len(params.causal_conv_channels)

	quantized_signal = np.mod(np.arange(1, input_width * 10), params.quantization_steps)
	print quantized_signal

	for rep in xrange(300):
		sum_loss = 0
		for train in xrange(50):
			# create batch
			input_batch, target_batch = create_batch(quantized_signal, batch_size, input_width, target_width)

			# convert to 1xW image whose #channels is equal to the quantization steps of audio
			# input_batch.shape = (BATCHSIZE, CHANNELS(=quantization_steps), HEIGHT(=1), WIDTH(=input_width))
			input_batch = data.onehot_pixel_image(input_batch, quantization_steps=params.quantization_steps)

			# training
			## causal block
			output = wavenet.forward_causal_block(input_batch)
			## remove causal padding
			output = wavenet.slice_1d(output, len(params.causal_conv_channels))
			## residual dilated conv block
			output, sum_skip_connections = wavenet.forward_residual_block(output)
			## remove unnecessary elements
			sum_skip_connections = wavenet.slice_1d(sum_skip_connections, sum_skip_connections.data.shape[3] - target_width)
			## softmax block
			## Note: do not apply F.softmax
			output = wavenet.forward_softmax_block(sum_skip_connections, softmax=False)
			## compute cross entroy
			loss = wavenet.cross_entropy(output, target_batch)
			## update weights
			wavenet.backprop(loss)

			sum_loss += float(loss.data)

		print sum_loss / 50.0
		wavenet.save(args.model_dir)



def main():
	receptive_field_milliseconds = 250
	train_audio()

if __name__ == '__main__':
	main()
