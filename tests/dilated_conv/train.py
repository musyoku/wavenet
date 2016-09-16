from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from args import args
from model import params, wavenet


input_size = 4
batchsize = 2
data = np.arange(batchsize * params.audio_channels * input_size).reshape((batchsize, params.audio_channels, 1, input_size)).astype(np.float32)
wavenet.forward_residual(data)
