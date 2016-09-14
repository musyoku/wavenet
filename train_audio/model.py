# -*- coding: utf-8 -*-
from args import args
from wavenet import WaveNet, Conf

conf = Conf()
conf.gpu_enabled = True if args.gpu_enabled == 1 else False

conf.learning_rate = 0.0003
conf.gradient_momentum = 0.9
conf.gradient_clipping = 10.0

conf.audio_channels = 256

conf.conv_kernel_sizes= [2, 2, 2]
conf.conv_out_channels= [256, 256, 256]

wavenet = WaveNet(conf)
wavenet.load(args.model_dir)