# -*- coding: utf-8 -*-
from args import args
from wavenet import WaveNet, Conf

conf = Conf()
conf.gpu_enabled = True if args.gpu_enabled == 1 else False

conf.audio_channels = 7

conf.residual_conv_kernel_size= 2
conf.residual_conv_channels = [3]

wavenet = WaveNet(conf)
wavenet.load(args.model_dir)