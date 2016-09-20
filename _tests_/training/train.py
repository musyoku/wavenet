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

	target_width = 5
	padded_input_width = 9
	batch_size = 5

	quantized_signal = np.mod(np.arange(1, 100), 6)
	quantized_signal = np.repeat(np.arange(0, 10), 100, axis=0)
	print quantized_signal

	for epoch in xrange(30):
		for step in xrange(100):
			padded_signal_batch, target_batch = create_batch(quantized_signal, batch_size, padded_input_width, target_width)
			
			padded_onehot_batch = data.onehot_pixel_image(padded_signal_batch, quantized_channels=params.quantization_steps)

			# print padded_signal_batch[0, -1]
			# print padded_onehot_batch[0, :, 0, -1]
			# print target_batch[0, -1]

			loss = wavenet.loss(padded_onehot_batch, target_batch)
			wavenet.backprop(loss)

		loss = float(loss.data)
		if loss > 0.3:
			print padded_signal_batch
			print target_batch



def main():
	receptive_field_milliseconds = 250
	train_audio()

if __name__ == '__main__':
	main()
