# -*- coding: utf-8 -*-
import json, os
from args import args
from wavenet import WaveNet, Params
from faster_wavenet import FasterWaveNet

# load params.json
try:
	os.mkdir(args.params_dir)
except:
	pass
filename = args.params_dir + "/{}".format(args.params_filename)
if os.path.isfile(filename):
	f = open(filename)
	dict = json.load(f)
	params = Params(dict)
	params.gpu_enabled = True if args.gpu_enabled == 1 else False
	wavenet = FasterWaveNet(params)
else:
	params = Params()
	params.gpu_enabled = True if args.gpu_enabled == 1 else False
	params.quantization_steps = 256
	params.sampling_rate = 8000

	params.causal_conv_no_bias = False
	params.causal_conv_filter_width = 2
	params.causal_conv_channels = [128]

	params.residual_conv_dilation_no_bias = True
	params.residual_conv_projection_no_bias = True
	params.residual_conv_filter_width = 3
	params.residual_conv_channels = [32, 32, 32, 32, 32, 32]
	params.residual_num_blocks = 2

	params.softmax_conv_no_bias = False
	params.softmax_conv_channels = [128, 256]

	params.learning_rate = 0.001
	params.gradient_momentum = 0.9
	params.weight_decay = 0.000001
	params.gradient_clipping = 10.0

	wavenet = FasterWaveNet(params)
	f = open(filename, "w")
	json.dump(params.to_dict(), f, indent=4)

params.dump()
wavenet.load(args.model_dir)