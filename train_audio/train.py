from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from model import params, wavenet
import data

def create_padded_batch(signal, batch_size, pos, shift, receptive_width, padded_width):
	# e.g.
	# input window:  0 - 100
	# target window: 1 - 101
	# therefore we crop 0 - 101
	crop = signal[pos * batch_size * padded_width + shift:(pos + 1) * batch_size * padded_width + shift + 1]
	x = crop[:-1]
	y = crop[1:]

	# insted of padding zero for the residual conv block, we pad with the actual signals
	padded_input_batch = x.reshape((batch_size, -1))

	# remove padding
	target_batch = y.reshape((batch_size, -1))[:,(padded_width - receptive_width):]

	return padded_input_batch, target_batch

def train_audio(
		filename, 
		batch_size=4,
		save_per_update=500,
		log_per_update=100,
		repeat=100
	):
	quantized_signal, sampling_rate = data.load_audio_file(filename, quantized_channels=params.audio_channels)

	# receptive field width for the top residual dilated conv layer
	# receptive field width is determined automatically when determining the depth of the residual dilated conv block
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
		
	# pad with zero
	quantized_signal = np.insert(quantized_signal, 0, np.zeros((padded_input_width,), dtype=np.int32), axis=0)

	pos_range = quantized_signal.size // (padded_input_width * batch_size)
	max_step = pos_range * padded_input_width

	for rep in xrange(repeat):
		for pos in xrange(pos_range):
			for shift in xrange(padded_input_width):
				# check if we can create batch
				if (pos + 1) * padded_input_width * batch_size + shift + 1 < quantized_signal.size:
					# create batch
					padded_input_batch, target_batch = create_padded_batch(quantized_signal, batch_size, pos, shift, target_width, padded_input_width)

					# convert to 1xW image whose channel is equal to quantized audio_channels
					# padded_x_batch.shape = (BATCHSIZE, CHANNELS(=audio channels), HEIGHT(=1), WIDTH(=receptive field))
					padded_x_batch = data.onehot_pixel_image(padded_input_batch, quantized_channels=params.audio_channels)

					# update weights
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
	train_audio("./wav_test/ring.wav")

if __name__ == '__main__':
	main()
