from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from model import params, wavenet
import data

def generate_audio(receptive_field_width_ms=25, sampling_rate=48000, generate_duration_sec=1):
	# e.g.
	# 48000 Hz * 0.25 = 12000 time steps (= 250 milliseconds receptive field)
	receptive_field_width_steps = int(sampling_rate * receptive_field_width_ms / 1000.0)

	# compute required input width
	batch_size = 1
	max_dilation = max(params.residual_conv_dilations)
	target_width = receptive_field_width_steps
	padded_input_width = receptive_field_width_steps + max_dilation

	# quantized signals generated by WaveNet
	generated_quantized_audio = np.zeros((padded_input_width, ), dtype=np.int32)

	for time_step in xrange(1, int(sampling_rate * generate_duration_sec)):
		# quantized signals in receptive field
		padded_quantized_x_batch = generated_quantized_audio[-padded_input_width:].reshape((1, -1))

		# convert to image
		padded_x_batch = data.onehot_pixel_image(padded_quantized_x_batch, quantized_channels=params.audio_channels)

		# generate next signal
		softmax = wavenet.forward_one_step(padded_x_batch, softmax=True, return_numpy=True)
		softmax = softmax[0, :, 0, -1]
		generated_quantized_signal = np.random.choice(np.arange(params.audio_channels), p=softmax)
		generated_quantized_audio = np.append(generated_quantized_audio, [generated_quantized_signal], axis=0)

		if time_step % 10 == 0:
			sys.stdout.write("\r generating {:.2f} msec / {:.2f} msec".format(time_step * 1000.0 / sampling_rate, generate_duration_sec * 1000.0))
			sys.stdout.flush()

	try:
		os.mkdir(args.generate_dir)
	except:
		pass
	filename = "{}/generated.wav".format(args.generate_dir)
	data.save_audio_file(filename, generated_quantized_audio, params.audio_channels, format="16bit_pcm", sampling_rate=sampling_rate)

def main():
	generate_audio(generate_duration_sec=args.generate_sec, sampling_rate=args.sampling_rate)

if __name__ == '__main__':
	main()
