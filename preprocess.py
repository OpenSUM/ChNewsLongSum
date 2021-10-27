import argparse
import os

parser = argparse.ArgumentParser(description='preprocess.py')

parser.add_argument('-dataset', required=True, help='cnn/cnndaily/lcsts')
parser.add_argument('-src', required=True, help='input dataset directory')
parser.add_argument('-dst', default=None, help='output dataset directory')
parser.add_argument('-vocab_size', default=None, help='vocab size')

config = parser.parse_args()

if config.dst is None:
    config.dst = os.path.join(config.src, 'output')

