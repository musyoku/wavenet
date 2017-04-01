# -*- coding: utf-8 -*-
import json, os
from args import args
from chainer import cuda
from wavenet import WaveNet, Params

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass
filename = args.model_dir + "/wavenet.json"
if os.path.isfile(filename):
	f = open(filename)
	dict = json.load(f)
	params = Params(dict)
	wavenet = WaveNet(params)
else:
	params = Params()
	params.quantization_steps = 10

	params.causal_conv_no_bias = True
	params.causal_conv_filter_width = 3
	params.causal_conv_channels = [32]

	params.residual_conv_dilation_no_bias = True
	params.residual_conv_projection_no_bias = True
	params.residual_conv_filter_width = 3
	params.residual_conv_channels = [16, 16]
	params.residual_num_blocks = 1

	params.softmax_conv_no_bias = False
	params.softmax_conv_channels = [24, 10]

	params.optimizer = "eve"
	params.learning_rate = 0.001
	params.momentum = 0.9
	params.weight_decay = 0.00001
	params.gradient_clipping = 10.0

	wavenet = WaveNet(params)
	f = open(filename, "w")
	json.dump(params.to_dict(), open(filename, "w"), indent=4)

params.dump()
wavenet.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	wavenet.to_gpu()
	