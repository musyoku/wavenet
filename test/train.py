from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from model import conf, wavenet


receptive_size = wavenet.input_receptive_size
batchsize = 2
data = np.ones((batchsize, conf.audio_channels, receptive_size, 1)).astype(np.float32)
print data.shape
print receptive_size
wavenet.forward_residual(data)