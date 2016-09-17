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
		batch_size=8,
		save_per_update=500,
		log_per_update=500,
		repeat=5
	):
	quantized_signal, sampling_rate = data.load_audio_file(filename, quantized_channels=params.audio_channels)
	receptive_steps = params.residual_conv_dilations[-1] * (params.residual_conv_kernel_width - 1)
	receptive_msec = int(receptive_steps * 1000.0 / sampling_rate)
	print "training", filename	
	print "	sampling rate:", sampling_rate, "[Hz]"
	print "	receptive field width:", receptive_msec, "[millisecond]"
	print "	receptive field width:", receptive_steps, "[time step]"

	# compute required input width
	max_dilation = max(params.residual_conv_dilations)
	target_width = receptive_steps
	padded_input_width = receptive_steps + max_dilation * params.residual_conv_kernel_width

	num_updates = 0
	total_updates = 0
	sum_loss = 0

	if padded_input_width * batch_size + 1 > quantized_signal.size:
		raise Exception("batch_size too large")

	pos_range = quantized_signal.size // (padded_input_width * batch_size)
	max_step = pos_range * padded_input_width

	for rep in xrange(repeat):
		for pos in xrange(pos_range):
			for shift in xrange(padded_input_width):
				# check if we can create batch
				if (pos + 1) * padded_input_width * batch_size + shift + 1 < quantized_signal.size:
					padded_input_batch, target_batch = create_signal_batch(quantized_signal, batch_size, pos, shift, target_width, padded_input_width)

					# convert to 1xW image whose channel is equal to quantized audio_channels
					padded_x_batch = data.onehot_pixel_image(padded_input_batch, quantized_channels=params.audio_channels)

					loss = wavenet.loss(padded_x_batch, target_batch)
					wavenet.backprop(loss)

					# logging
					current_step = shift + pos * padded_input_width
					sum_loss += float(loss.data)
					num_updates += 1
					total_updates += 1
					if num_updates == log_per_update:
						print "{}/{} ({}/{}) loss: {:.6f}".format(rep + 1, repeat, current_step + 1, max_step, sum_loss / float(log_per_update))
						num_updates = 0
						sum_loss = 0
					if total_updates % save_per_update == 0:
						wavenet.save(dir=args.model_dir)

	wavenet.save(dir=args.model_dir)

def main():
	train_audio("./wav_test/voice.wav")

if __name__ == '__main__':
	main()
