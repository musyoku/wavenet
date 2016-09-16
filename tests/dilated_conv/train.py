from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from args import args
from model import params, wavenet


input_size = 4
batchsize = 2
data = np.ones((batchsize, params.audio_channels, input_size, 1)).astype(np.float32)
print data.shape
print input_size
wavenet.forward_residual(data)
