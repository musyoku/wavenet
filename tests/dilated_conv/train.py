from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from args import args
from model import params, wavenet

input_size = 5
batchsize = 3
data = np.arange(1, batchsize * params.audio_channels * input_size + 1).reshape((batchsize, params.audio_channels, 1, input_size)).astype(np.float32)
wavenet.forward_residual(data)
