from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from model import conf, wavenet

def load_audio_file(filename, format="16bit_pcm"):
	sr, signal = wavfile.read(filename)
	print signal
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
	print signal
	# apply mu-law companding transformation (ITU-T, 1988)
	mu = conf.audio_channels - 1
	signal = np.sign(signal) * np.log(1 + mu * np.absolute(signal)) / np.log(1 + mu)
	# quantize
	signal = (signal * mu).astype(int)
	return signal

def train_audio(filename):
	signal = load_audio_file(filename)

def main():
	train_audio("./wav/kamiya_0.wav")

if __name__ == '__main__':
	main()
