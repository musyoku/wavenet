from scipy.io import wavfile
import numpy as np
import os, sys, time
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from model import params, wavenet
import data

def generate_audio(sampling_rate=48000, generate_sec=1, remove_silence_frames=False):
	# compute required input width
	num_layers = len(params.residual_conv_channels)
	receptive_steps_per_unit = params.residual_conv_filter_width ** num_layers
	receptive_steps = (receptive_steps_per_unit - 1) * params.residual_num_blocks + 1
	input_width = receptive_steps
	# add paddings of causal conv block
	input_width += len(params.causal_conv_channels)

	# pad with silence signals
	generated_signals = np.full((input_width, ), 127, dtype=np.int32)

	start_time = time.time()
	for time_step in xrange(1, int(sampling_rate * generate_sec)):
		# signals in receptive field
		input_signals = generated_signals[-input_width:].reshape((1, -1))

		# convert to image
		input_signals = data.onehot_pixel_image(input_signals, quantization_steps=params.quantization_steps)

		# generate next signal
		if args.fast:
			softmax = wavenet._forward_one_step(input_signals, apply_softmax=True, as_numpy=True)
		else:
			softmax = wavenet.forward_one_step(input_signals, apply_softmax=True, as_numpy=True)


		softmax = softmax[0, :, 0, -1]
		signal = np.random.choice(np.arange(params.quantization_steps), p=softmax)

		if signal == 127 and remove_silence_frames:
			pass
		else:
			generated_signals = np.append(generated_signals, [signal], axis=0)

		if time_step % 10 == 0:
			sys.stdout.write("\rgenerating {:.2f} msec / {:.2f} msec".format(time_step * 1000.0 / sampling_rate, generate_sec * 1000.0))
			sys.stdout.flush()

	print "\ndone in {:.3f} sec".format(time.time() - start_time)

	# remove paddings
	generated_signals = generated_signals[input_width:]

	try:
		os.mkdir(args.output_dir)
	except:
		pass

	filename = "{}/generated.wav".format(args.output_dir)
	data.save_audio_file(filename, generated_signals, params.quantization_steps, format="16bit_pcm", sampling_rate=sampling_rate)


if __name__ == '__main__':
	generate_audio(generate_sec=args.seconds, sampling_rate=params.sampling_rate)