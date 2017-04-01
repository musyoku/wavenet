# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu_device", type=int, default=0)
parser.add_argument("-w", "--wav-dir", type=str, default="wav")
parser.add_argument("-m", "--model-dir", type=str, default="model")

# generation
parser.add_argument("-o", "--out_dir", type=str, default="generated_audio")
parser.add_argument("-s", "--generate_sec", type=float, default=1.0)
parser.add_argument("--use_faster_wavenet", action="store_true", default=False)

# seed
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()