from scipy.io import wavfile
import os, sys
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from model import conf, wavenet