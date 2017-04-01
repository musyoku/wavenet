# -*- coding: utf-8 -*-
import json, os
from args import args
from chainer import cuda
from wavenet import WaveNet, Params
from faster_wavenet import FasterWaveNet

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass
filename = args.model_dir + "/wavenet.json"
if os.path.isfile(filename):
	print "loading", filename
	f = open(filename)
	try:
		dict = json.load(f)
		params = Params(dict)
	except:
		raise Exception("could not load {}".format(filename))
else:
	params = Params()
	params.quantization_steps = 256
	params.sampling_rate = 8000

	params.causal_conv_no_bias = True
	params.causal_conv_filter_width = 2
	params.causal_conv_channels = [256]

	params.residual_conv_dilation_no_bias = True
	params.residual_conv_projection_no_bias = True
	params.residual_conv_filter_width = 2
	params.residual_conv_channels = [128, 128, 128, 128, 128, 128, 128, 128, 128]
	params.residual_num_blocks = 1

	params.softmax_conv_no_bias = False
	params.softmax_conv_channels = [256, 256]

	params.optimizer = "eve"
	params.momentum = 0.9
	params.weight_decay = 0
	params.gradient_clipping = 1.0
	
	with open(filename, "w") as f:
		json.dump(params.to_dict(), f, indent=4)


if args.fast:
	wavenet = FasterWaveNet(params)
else:
	wavenet = WaveNet(params)

params.dump()
wavenet.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	wavenet.to_gpu()