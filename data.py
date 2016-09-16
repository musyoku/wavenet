from scipy.io import wavfile
import numpy as np

def load_audio_file(filename, channels=256, format="16bit_pcm"):
	sampling_rate, signal = wavfile.read(filename)
	# discard R channel to convert to mono
	signal = signal[:, 0].astype(float)
	# normalize to -1 ~ 1
	if format == "16bit_pcm":
		max = 1<<15
	elif format == "32bit_pcm":
		max = 1<<31
	elif format == "8bit_pcm":
		max = 1<<8 - 1
	signal /= max
	# mu-law companding transformation (ITU-T, 1988)
	mu = channels - 1
	signal = np.sign(signal) * np.log(1 + mu * np.absolute(signal)) / np.log(1 + mu)
	# quantize
	quantized_signal = (np.clip(signal * 0.5 + 0.5, 0, 1) * mu).astype(int)
	return quantized_signal, sampling_rate

def save_audio_file(filename, signals_1d, channels, format="16bit_pcm", sampling_rate=48000):
	signals_1d = signals_1d.astype(float)
	signals_1d = (signals_1d / channels - 0.5) * 2.0

	# inv mu-law companding transformation (ITU-T, 1988)
	mu = channels - 1
	power = np.power(np.full(signals_1d.shape, 1 + mu, dtype=np.float32), np.absolute(signals_1d))
	signals_1d = np.sign(signals_1d) * (1.0 / mu) * (power - 1)

	if format == "16bit_pcm":
		max = 1<<15
	elif format == "32bit_pcm":
		max = 1<<31
	elif format == "8bit_pcm":
		max = 1<<8 - 1
	signals_1d *= max

	audio = signals_1d.reshape((-1, 1)).astype(int)
	audio = np.repeat(audio, 2, axis=1)

	wavfile.write(filename, sampling_rate, audio)

# convert signal to 1xW image
def onehot_pixel_image(quantized_signal_batch, channels=256):
	batchsize = quantized_signal_batch.shape[0]
	width = quantized_signal_batch.shape[1]
	image = np.zeros((batchsize * width, channels), dtype=np.float32)
	image[np.arange(batchsize * width), quantized_signal_batch.reshape((1, -1))] = 1
	image = image.reshape((batchsize, width, channels, 1))
	image = image.transpose((0, 2, 3, 1))
	return image
