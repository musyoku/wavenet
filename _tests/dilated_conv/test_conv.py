from scipy.io import wavfile
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from wavenet import params, wavenet

in_channels = 4
out_channels = 3

initial_w = np.random.normal(scale=std, size=shape_w)

l0 = DilatedConvolution1D(n_in, n_out, ksize, 
	kernel_width=2,
	dilation=1,
	stride=1, nobias=True, initialW=initial_w)
l0.kernel_width = kernel_width
l0.dilation = dilation