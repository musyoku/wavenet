from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from args import args
from model import params, wavenet
import data

def create_padded_batch(signal, batch_size, pos, shift, receptive_width, padded_width):
	crop = signal[pos * batch_size * padded_width + shift:(pos + 1) * batch_size * padded_width + shift + 1]
	x = crop[:-1]
	y = crop[1:]
	input_batch = x.reshape((batch_size, -1))
	target_batch = y.reshape((batch_size, -1))[:,(padded_width - receptive_width):]
	return input_batch, target_batch


def train_audio():

	target_width = 4
	padded_input_width = 8 + 3 + 1
	batch_size = 1

	quantized_signal = np.mod(np.arange(1, padded_input_width * batch_size * 4), 6)
	print quantized_signal

	for rep in xrange(30):
		for pos in xrange(quantized_signal.size // (padded_input_width * batch_size)):
			for shift in xrange(padded_input_width):
				if (pos + 1) * padded_input_width * batch_size + shift + 1 < quantized_signal.size:
					padded_signal_batch, target_batch = create_padded_batch(quantized_signal, batch_size, pos, shift, target_width, padded_input_width)
					
					padded_onehot_batch = data.onehot_pixel_image(padded_signal_batch, quantized_channels=params.audio_channels)

					# print padded_signal_batch[0, -1]
					# print padded_onehot_batch[0, :, 0, -1]
					# print target_batch[0, -1]

					# causal block
					print "[#1]"
					output = wavenet.forward_causal_block(padded_onehot_batch)
					output = wavenet.slice_1d(output, 1)
					output, sum_skip_connections = wavenet.forward_residual_block(output)

					print "[#2]"
					output = wavenet.forward_causal_block(padded_onehot_batch[:,:,:,-9:])
					output = wavenet.slice_1d(output, 1)
					output, sum_skip_connections = wavenet.forward_residual_block(output)

					raise Exception()


					# loss = wavenet.loss(padded_onehot_batch, target_batch)
					# wavenet.backprop(loss)

		print float(loss.data)

	wavenet.save(args.model_dir)


def main():
	receptive_field_milliseconds = 250
	train_audio()

if __name__ == '__main__':
	main()
