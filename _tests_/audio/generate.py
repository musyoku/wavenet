from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import data

def main():
	filename = "../../train_audio/wav_test/ring.wav"
	quantized_signal, sampling_rate = data.load_audio_file(filename, quantized_channels=256)
	filename = "generated.wav"
	data.save_audio_file(filename, quantized_signal, 256, format="16bit_pcm", sampling_rate=sampling_rate)

if __name__ == '__main__':
	main()
