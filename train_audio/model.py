# -*- coding: utf-8 -*-
import json, os
from args import args
from wavenet import WaveNet, Params

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
	wavenet = WaveNet(params)
else:
	params = Params()
	params.audio_channels = 256

	params.causal_conv_no_bias = True
	params.causal_conv_kernel_width = 2
	params.causal_conv_apply_batchnorm = True
	params.causal_conv_batchnorm_before_conv = True
	params.causal_conv_channels = [128]

	params.residual_conv_dilation_no_bias = True
	params.residual_conv_projection_no_bias = True
	params.residual_conv_kernel_width = 2
	params.residual_conv_apply_batchnorm = True
	params.residual_conv_batchnorm_before_conv = True
	params.residual_conv_channels = [32, 32, 32, 32]
	params.residual_conv_dilations = [1, 2, 4, 8]

	params.softmax_conv_no_bias = False
	params.softmax_wscale = 0.01
	params.softmax_conv_kernel_width = 2
	params.softmax_conv_apply_batchnorm = True
	params.softmax_conv_batchnorm_before_conv = True
	params.softmax_conv_channels = [128, 256]

	params.learning_rate = 0.001
	params.gradient_momentum = 0.9
	params.weight_decay = 0.00001
	params.gradient_clipping = 10.0

	wavenet = WaveNet(params)
	f = open(filename, "w")
	json.dump(params.to_dict(), f, indent=4)

params.gpu_enabled = True if args.gpu_enabled == 1 else False
params.dump()
wavenet.load(args.model_dir)