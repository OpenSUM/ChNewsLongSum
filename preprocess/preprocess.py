import argparse
import json
import os

from subword_nmt.learn_bpe import learn_bpe

parser = argparse.ArgumentParser(description='preprocess.py')

# parser.add_argument('-dataset', required=True, help='cnn/cnndaily/lcsts')
parser.add_argument('-dataset', required=True, help='src jsonl file')
# parser.add_argument('-dst', required=True, help='dst jsonl file')
# parser.add_argument('-vocab_size', default=None, help='vocab size')

config = parser.parse_args()

# if config.dst is None:
#     config.dst = os.path.join(config.src, 'output')

if __name__ == '__main__':
    with open(config.dataset, 'r', encoding='utf8') as f:
        data = [json.loads(line.strip()) for line in f]
    texts = []
    for js in data:
        texts.append(js['text'])
        texts.append(js['abstract'])
    with open('./train.codec', 'w', encoding='utf8') as f:
        learn_bpe(infile=texts, outfile=f, num_symbols=64000, verbose=True)
