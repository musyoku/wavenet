from scipy.io import wavfile
import numpy as np
import os, sys
from chainer import Variable
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from wavenet import DilatedConvolution1D

in_channels = 4
out_channels = 3
filter_width = 4
width = 10

x = np.mod(np.arange(0, in_channels * width), 5).reshape((1, in_channels, 1, width)).astype(np.float32)
print x

initial_w = np.ones((out_channels, in_channels, filter_width, 1), dtype=np.float32)

l0 = DilatedConvolution1D(in_channels, out_channels,
	ksize=(filter_width, 1), 
	filter_width=filter_width,
	layer_index=1,
	stride=1, nobias=True, initialW=initial_w)

x = Variable(x)
out = l0(x)
print out.data
