from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.split(os.getcwd())[0])
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

def train_audio(
		filename, 
		batch_size=16,
		save_per_update=500,
		log_per_update=50,
		epochs=100
	):
	quantized_signal, sampling_rate = data.load_audio_file(filename, quantized_channels=params.quantization_steps)

	residual_conv_dilations = []
	dilation = 1
	for _ in params.residual_conv_channels:
		residual_conv_dilations.append(dilation)
		dilation *= 2
	max_dilation = max(residual_conv_dilations)

	# receptive field width for the top residual dilated conv layer
	# receptive field width is determined automatically when determining the depth of the residual dilated conv block
	receptive_steps = max_dilation * params.residual_conv_kernel_width
	receptive_msec = int(receptive_steps * 1000.0 / sampling_rate)
	target_width = receptive_steps
	# compute required input width
	padded_input_width = target_width * 2 - 1
	# padding for causal conv block
	padded_input_width += len(params.causal_conv_channels)

	print "training", filename	
	print "	sampling rate:", sampling_rate, "[Hz]"
	print "	receptive field width:", receptive_msec, "[millisecond]"
	print "	receptive field width:", receptive_steps, "[time step]"
	print "	batch_size:", batch_size
	print "	learning_rate:", params.learning_rate

	num_updates = 0
	total_updates = 0
	sum_loss = 0

	if padded_input_width * batch_size + 1 > quantized_signal.size:
		raise Exception("batch_size too large")

	# pad with zero
	quantized_signal = np.insert(quantized_signal, 0, np.zeros((padded_input_width,), dtype=np.int32), axis=0)

	max_batches = int((quantized_signal.size - padded_input_width) / float(batch_size))

	for epoch in xrange(1, epochs + 1):
		print "epoch: {}/{}".format(epoch, epochs)
		for batch_index in xrange(1, max_batches + 1):
			# create batch
			padded_input_batch, target_batch = create_batch(quantized_signal, batch_size, padded_input_width, target_width)

			# convert to 1xW image whose #channels is equal to the quantization steps of audio
			# padded_x_batch.shape = (BATCHSIZE, CHANNELS(=quantization_steps), HEIGHT(=1), WIDTH(=padded_input_width))
			padded_x_batch = data.onehot_pixel_image(padded_input_batch, quantized_channels=params.quantization_steps)

			# compute output
			output = wavenet.forward_causal_block(padded_x_batch)
			# remove causal padding
			output = wavenet.slice_1d(output, len(params.causal_conv_channels))
			output, sum_skip_connections = wavenet.forward_residual_block(output)
			# remove padding
			sum_skip_connections = wavenet.slice_1d(sum_skip_connections, output.data.shape[3] - target_width)
			# do not apply F.softmax
			output = wavenet.forward_softmax_block(sum_skip_connections, softmax=False)
			# compute cross entroy
			loss = wavenet.cross_entropy(output, target_batch)
			# update weights
			wavenet.backprop(loss)

			# logging
			sum_loss += float(loss.data)
			total_updates += 1
			if batch_index % log_per_update == 0:
				print "	batch: {}/{} loss: {:.6f}".format(batch_index, max_batches, sum_loss / float(log_per_update))
				sum_loss = 0

			# save the model
			if total_updates % save_per_update == 0:
				wavenet.save(dir=args.model_dir)

		wavenet.save(dir=args.model_dir)
	wavenet.save(dir=args.model_dir)

def main():
	train_audio("./wav_test/famima.wav")

if __name__ == '__main__':
	main()
