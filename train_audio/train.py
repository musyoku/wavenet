from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.split(os.getcwd())[0])
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
		filename, 
		batch_size=4,
		save_per_update=500,
		repeat=100
	):
	# e.g.
	# 48000 Hz * 0.25 = 12000 time steps (= 250 milliseconds receptive field)
	quantized_signal, sampling_rate = data.load_audio_file(filename, channels=params.audio_channels)
	receptive_field_width_steps = params.residual_conv_dilations[-1]
	receptive_field_width_ms = int(receptive_field_width_steps * 1000.0 / sampling_rate)
	print "training", filename	
	print "	sampling rate:", sampling_rate, "[Hz]"
	print "	receptive field width:", receptive_field_width_ms, "[millisecond]"
	print "	receptive field width:", receptive_field_width_steps, "[time step]"

	# compute required input width
	max_dilation = max(params.residual_conv_dilations)
	target_width = receptive_field_width_steps
	padded_input_width = receptive_field_width_steps + max_dilation

	num_updates = 0
	total_updates = 0
	sum_loss = 0

	if padded_input_width * batch_size + 1 > quantized_signal.size:
		raise Exception("batch_size too large")

	for rep in xrange(repeat):
		for pos in xrange(quantized_signal.size // (padded_input_width * batch_size)):
			for shift in xrange(padded_input_width):
				# check if we can create batch
				if (pos + 1) * padded_input_width * batch_size + shift + 1 < quantized_signal.size:
					# print pos, shift
					padded_quantized_input_signal_batch, quantized_target_signal_batch = create_signal_batch(quantized_signal, batch_size, pos, shift, target_width, padded_input_width)
					# print padded_quantized_input_signal_batch
					# print quantized_target_signal_batch
					# convert to 1xW image whose channel is equal to audio_channels
					padded_x_batch = data.onehot_pixel_image(padded_quantized_input_signal_batch, channels=params.audio_channels)
					loss = wavenet.loss(padded_x_batch, quantized_target_signal_batch)
					wavenet.backprop(loss)

					# logging
					sum_loss += float(loss.data)
					num_updates += 1
					total_updates += 1
					if num_updates == 50:
						print "loss: {:.6f}".format(sum_loss / 50.0)
						num_updates = 0
						sum_loss = 0
					if total_updates % save_per_update == 0:
						wavenet.save(dir=args.model_dir)

	wavenet.save(dir=args.model_dir)

def main():
	train_audio("./wav_test/voice.wav")

if __name__ == '__main__':
	main()
