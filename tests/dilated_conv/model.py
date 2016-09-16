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
	params.gpu_enabled = True if args.gpu_enabled == 1 else False
	params.audio_channels = 3
	params.residual_conv_kernel_width = 2
	params.residual_conv_channels = [3]
	params.softmax_conv_channels = [3]
	params.causal_conv_channels = [3, 3, 3]
	params.residual_conv_dilations = [2]
	params.causal_conv_apply_batchnorm = False
	params.residual_conv_apply_batchnorm = False
	params.softmax_conv_apply_batchnorm = False

	wavenet = WaveNet(params)
	f = open(filename, "w")
	json.dump(params.to_dict(), f, indent=4)

params.dump()
wavenet.load(args.model_dir)