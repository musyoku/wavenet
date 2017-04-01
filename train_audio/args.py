# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu_device", type=int, default=0)
parser.add_argument("-w", "--wav-dir", type=str, default="wav")
parser.add_argument("-m", "--model-dir", type=str, default="model")

# generation
parser.add_argument("-o", "--output_dir", type=str, default="generated_audio")
parser.add_argument("-s", "--seconds", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=0.00001, help="learning_rate")
parser.add_argument("--fast", action="store_true", default=False)

# seed
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()