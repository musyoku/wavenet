from scipy.io import wavfile
import numpy as np
import os, sys, time, chainer
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from model import params, wavenet
import data

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def create_batch(signal, batch_size, input_width, target_width):
	indecis = np.random.randint(0, signal.size - target_width - input_width - 1, size=batch_size)
	input_batch = np.empty((batch_size, input_width + target_width), dtype=np.int32)
	target_batch = np.empty((batch_size, target_width), dtype=np.int32)
	for n in xrange(batch_size):
		start = indecis[n]
		input_batch[n] = signal[start:start + input_width + target_width]
		target_batch[n] = signal[start + input_width + 1:start + input_width + target_width + 1]
	return input_batch, target_batch

def train_audio(
		filename, 
		batch_size=16,
		train_width=16,
		repeat=1000,
	):

	# load audio data
	path_to_file = args.wav_dir + "/" + filename
	signals, sampling_rate = data.load_audio_file(path_to_file, quantization_steps=params.quantization_steps)

	# receptive width
	num_layers = len(params.residual_conv_channels)
	receptive_width_per_unit = params.residual_conv_filter_width ** num_layers
	receptive_width = (receptive_width_per_unit - 1) * params.residual_num_blocks + 1
	receptive_msec = int(receptive_width * 1000.0 / sampling_rate)

	# receptive field width
	input_width = receptive_width
	# padding for causal conv block
	input_width += len(params.causal_conv_channels)

	# for logging
	num_updates = 0
	total_updates = 0
	sum_loss = 0
	prev_average_loss = None

	# pad with silence signals
	signals = np.insert(signals, 0, np.full((input_width,), 127, dtype=np.int32), axis=0)

	with chainer.using_config("train", True):
		for batch_index in xrange(0, repeat):
			# create batch
			input_batch, target_batch = create_batch(signals, batch_size, input_width, train_width)

			# convert to 1xW image whose #channels is equal to the quantization steps of audio
			# input_batch.shape = (BATCHSIZE, CHANNELS(=quantization_steps), HEIGHT(=1), WIDTH(=input_width))
			input_batch = data.onehot_pixel_image(input_batch, quantization_steps=params.quantization_steps)

			# training
			## causal block
			output = wavenet.forward_causal_block(input_batch)
			## remove causal padding
			# output = wavenet.slice_1d(output, len(params.causal_conv_channels))
			## residual dilated conv block
			output, sum_skip_connections = wavenet.forward_residual_block(output)
			## remove unnecessary elements
			output = wavenet.slice_1d(output, output.data.shape[3] - train_width)
			sum_skip_connections = wavenet.slice_1d(sum_skip_connections, sum_skip_connections.data.shape[3] - train_width)
			## softmax block
			## Note: do not apply F.softmax
			output = wavenet.forward_softmax_block(sum_skip_connections, apply_softmax=False)
			## compute cross entroy
			loss = wavenet.cross_entropy(output, target_batch)
			## update weights
			wavenet.backprop(loss)

			# logging
			sum_loss += float(loss.data)
			total_updates += 1

			if batch_index % 10 == 0:
				sys.stdout.write("\r	{} - {} width; {}/{}".format(stdout.BOLD + filename + stdout.END, signals.size, batch_index, repeat))
				sys.stdout.flush()

	wavenet.save(args.model_dir)
	return sum_loss

def main():
	np.random.seed(args.seed)
	wavenet.update_laerning_rate(args.lr)

	files = []
	fs = os.listdir(args.wav_dir)
	for fn in fs:
 		# filter out non-wav files
 		if fn.endswith('.wav'):
 			print "loading", fn
 			files.append(fn)

	# compute receptive field width
	num_layers = len(params.residual_conv_channels)
	receptive_width_per_unit = params.residual_conv_filter_width ** num_layers
	receptive_width = (receptive_width_per_unit - 1) * params.residual_num_blocks + 1
	receptive_msec = int(receptive_width * 1000.0 / params.sampling_rate)
	print "receptive field width:", receptive_msec, "[millisecond]"
	print "receptive field width:", receptive_width, "[step]"

	batch_size = 16
	train_width = 500
	max_epoch = 2000
	start_time = time.time()
	print "files: {} batch_size: {} train_width: {}".format(len(files), batch_size, train_width)

	for epoch in xrange(1, max_epoch):
		average_loss = 0
		for i, filename in enumerate(files):
			# train
			loss = train_audio(filename, 
				batch_size=batch_size,
				train_width=train_width,
				repeat=500
			)
			average_loss += loss

		average_loss /= len(files)
		sys.stdout.write(stdout.CLEAR)
		sys.stdout.write("\repoch: {} - {:.4e} loss - {} min\n".format(epoch, average_loss, int((time.time() - start_time) / 60)))
		sys.stdout.flush()

		wavenet.save(args.model_dir)


if __name__ == '__main__':
	main()
