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
	print "loading", filename
	f = open(filename)
	try:
		dict = json.load(f)
		params = Params(dict)
	except:
		raise Exception("could not load {}".format(filename))

	params.gpu_enabled = True if args.gpu_enabled == 1 else False

	if args.use_faster_wavenet:
		wavenet = FasterWaveNet(params)
	else:
		wavenet = WaveNet(params)
else:
	params = Params()
	params.quantization_steps = 256
	params.sampling_rate = 8000

	params.causal_conv_no_bias = False
	params.causal_conv_filter_width = 2
	params.causal_conv_channels = [32]

	params.residual_conv_dilation_no_bias = True
	params.residual_conv_projection_no_bias = True
	params.residual_conv_filter_width = 2
	params.residual_conv_channels = [16, 16, 16, 16, 16]
	params.residual_num_blocks = 2

	params.softmax_conv_no_bias = False
	params.softmax_conv_channels = [32, 128, 256]

	params.learning_rate = 0.001
	params.gradient_momentum = 0.9
	params.weight_decay = 0.000001
	params.gradient_clipping = 10.0
	
	params.gpu_enabled = True if args.gpu_enabled == 1 else False

	if args.use_faster_wavenet:
		wavenet = FasterWaveNet(params)
	else:
		wavenet = WaveNet(params)

	with open(filename, "w") as f:
		json.dump(params.to_dict(), f, indent=4)

params.dump()
wavenet.load(args.model_dir)