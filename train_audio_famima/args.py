# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_enabled", type=int, default=1)
parser.add_argument("--wav_dir", type=str, default="wav")
parser.add_argument("--model_dir", type=str, default="model")

# params
parser.add_argument("--params_dir", type=str, default="params")
parser.add_argument("--params_filename", type=str, default="params.json")

# generation
parser.add_argument("--generate_dir", type=str, default="generated_audio")
parser.add_argument("--generate_sec", type=float, default=1.0)
parser.add_argument("--use_faster_wavenet", type=str, default=None)

# seed
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()

if args.use_faster_wavenet == "True" or args.use_faster_wavenet == "true":
	args.use_faster_wavenet = True
else:
	args.use_faster_wavenet = False