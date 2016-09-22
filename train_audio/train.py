from scipy.io import wavfile
import numpy as np
import os, sys, time
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
		learnable_steps=16,
		save_per_update=500,
		train_steps_ratio=0.05,
	):

	# load audio data
	path_to_file = args.wav_dir + "/" + filename
	quantized_signal, sampling_rate = data.load_audio_file(path_to_file, quantization_steps=params.quantization_steps)

	# compute receptive field width
	num_layers = len(params.residual_conv_channels)
	receptive_steps_per_unit = params.residual_conv_filter_width ** num_layers
	receptive_steps = (receptive_steps_per_unit - 1) * params.residual_num_blocks + 1
	receptive_msec = int(receptive_steps * 1000.0 / sampling_rate)
	target_width = learnable_steps
	input_width = receptive_steps
	# to compute all learnable targets
	input_width += learnable_steps - 1
	## padding for causal conv block
	input_width += len(params.causal_conv_channels)

	# for logging
	num_updates = 0
	total_updates = 0
	sum_loss_epoch = 0
	sum_loss = 0
	start_time = time.time()
	prev_averate_loss = None
	max_batches = max(int((quantized_signal.size - input_width) / float(batch_size) * train_steps_ratio), 1)

	# print "training", filename	
	# print "	sampling rate:", sampling_rate, "[Hz]"
	# print "	length:", quantized_signal.size, "[step]"
	# print "	batch_size:", batch_size
	# print "	learnable_steps:", learnable_steps

	# pad with zero
	quantized_signal = np.insert(quantized_signal, 0, np.zeros((input_width,), dtype=np.int32), axis=0)

	sum_loss_epoch = 0
	sum_loss = 0
	start_time = time.time()
	for batch_index in xrange(1, max_batches + 1):
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

		# logging
		loss = float(loss.data)
		sum_loss_epoch += loss
		sum_loss += loss
		total_updates += 1

		# save the model
		if total_updates % save_per_update == 0:
			wavenet.save(dir=args.model_dir)

	wavenet.save(dir=args.model_dir)
	average_loss = sum_loss / float(max_batches)
	sys.stdout.flush()

	return average_loss

def main():
	final_learning_rate = 0.00000001
	np.random.seed(args.seed)

	files = []
	fs = os.listdir(args.wav_dir)
	for fn in fs:
		print "loading", fn
		files.append(fn)

	# compute receptive field width
	num_layers = len(params.residual_conv_channels)
	receptive_steps_per_unit = params.residual_conv_filter_width ** num_layers
	receptive_steps = (receptive_steps_per_unit - 1) * params.residual_num_blocks + 1
	receptive_msec = int(receptive_steps * 1000.0 / params.sampling_rate)
	print "receptive field width:", receptive_msec, "[millisecond]"
	print "receptive field width:", receptive_steps, "[step]"

	current_learning_rate = params.learning_rate
	prev_averate_loss = None
	max_epoch = 2000
	for epoch in xrange(1, max_epoch):
		average_loss = 0
		for i, filename in enumerate(files):
			sys.stdout.write("\repoch: {}/{} file: {}/{} {}".format(epoch, max_epoch, i+1, len(files), filename))
			sys.stdout.flush()
			# train
			loss = train_audio(filename, 
				batch_size=16,
				learnable_steps=32,
				save_per_update=500,
				train_steps_ratio=0.05
			)
			average_loss += loss

		average_loss /= len(files)
		sys.stdout.write("\repoch: {}/{} loss: {:.3f}								".format(epoch, max_epoch, average_loss))
		sys.stdout.flush()
		sys.stdout.write("\n")

		# anneal learning rate
		if prev_averate_loss is not None:
			if average_loss > prev_averate_loss and current_learning_rate > final_learning_rate:
				current_learning_rate *= 0.1
				wavenet.update_laerning_rate(current_learning_rate)
				print "learning rate annealed to", current_learning_rate

		prev_averate_loss = average_loss
		wavenet.save(dir=args.model_dir)


if __name__ == '__main__':
	main()
