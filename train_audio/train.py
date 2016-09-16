from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from model import params, wavenet
import data

def train_audio(filename, receptive_field_width_ms=25):
	# e.g.
	# 48000 Hz * 0.25 = 12000 time steps (= 250 milliseconds receptive field)
	signal, sampling_rate = data.load_audio_file(filename, channels=params.audio_channels)
	receptive_field_width_steps = int(sampling_rate * receptive_field_width_ms / 1000.0)
	print "training", filename	
	print "	sampling rate:", sampling_rate, "[Hz]"
	print "	receptive field width:", receptive_field_width_ms, "[millisecond]"
	print "	receptive field width:", receptive_field_width_steps, "[time step]"

def main():
	receptive_field_milliseconds = 250
	train_audio("./wav/kamiya_0.wav")

if __name__ == '__main__':
	main()
