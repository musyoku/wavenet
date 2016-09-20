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

	target_width = 4
	padded_input_width = 8 + 3 + 1
	batch_size = 8

	quantized_signal = np.mod(np.arange(1, padded_input_width * batch_size * 4), 6)
	# pad with zero
	quantized_signal = np.insert(quantized_signal, 0, np.ones((padded_input_width,), dtype=np.int32), axis=0)
	print quantized_signal

	for rep in xrange(50):
		for step in xrange(10):
			padded_signal_batch, target_batch = create_batch(quantized_signal, batch_size, padded_input_width, target_width)
			
			padded_onehot_batch = data.onehot_pixel_image(padded_signal_batch, quantized_channels=params.quantization_steps)

			# print padded_signal_batch[0, -1]
			# print padded_onehot_batch[0, :, 0, -1]
			# print target_batch[0, -1]

			output = wavenet.forward_causal_block(padded_onehot_batch)
			output = wavenet.slice_1d(output, 1)
			output, sum_skip_connections = wavenet.forward_residual_block(output)
			sum_skip_connections = wavenet.slice_1d(sum_skip_connections, output.data.shape[3] - target_width)
			output = wavenet.forward_softmax_block(sum_skip_connections, softmax=False)
			loss = wavenet.cross_entropy(output, target_batch)
			wavenet.backprop(loss)

		loss = float(loss.data)
		print loss

	wavenet.save(args.model_dir)


def main():
	receptive_field_milliseconds = 250
	train_audio()

if __name__ == '__main__':
	main()
