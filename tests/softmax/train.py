from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from args import args
from model import params, wavenet
import data

def create_signal_batch(signal, batch_size, pos, shift, receptive_width, padded_width):
	crop = signal[pos * batch_size * padded_width + shift:(pos + 1) * batch_size * padded_width + shift + 1]
	x = crop[:-1]
	y = crop[1:]
	input_batch = x.reshape((batch_size, -1))
	target_batch = y.reshape((batch_size, -1))[:,(padded_width - receptive_width):]
	return input_batch, target_batch

def train_audio(
		receptive_field_width_ms=25,
		batch_size=3
	):

	target_width = 3
	padded_input_width = 6

	quantized_signal = np.mod(np.arange(1, padded_input_width * batch_size * 4), 3)
	print quantized_signal

	for pos in xrange(quantized_signal.size // (padded_input_width * batch_size)):
		for shift in xrange(padded_input_width):
			# check if we can create batch
			if (pos + 1) * padded_input_width * batch_size + shift + 1 < quantized_signal.size:
				# print pos, shift
				padded_quantized_input_signal_batch, quantized_target_signal_batch = create_signal_batch(quantized_signal, batch_size, pos, shift, target_width, padded_input_width)
				print padded_quantized_input_signal_batch
				# print quantized_target_signal_batch
				# convert to 1xW image whose channel is equal to audio_channels
				padded_x_batch = data.onehot_pixel_image(padded_quantized_input_signal_batch, channels=params.audio_channels)
				loss = wavenet.loss(padded_x_batch, quantized_target_signal_batch)
				wavenet.backprop(loss)
				print loss.data

def main():
	receptive_field_milliseconds = 250
	train_audio()

if __name__ == '__main__':
	main()
