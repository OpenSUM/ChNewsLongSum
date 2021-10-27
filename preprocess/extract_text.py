import json

files = ['train', 'eval', 'test']
for filename in files:
    in_file = './preprocessed/{}.jsonl'.format(filename)
    src_file = './final/{}.src.token'.format(filename)
    dst_file = './final/{}.dst.token'.format(filename)
    with open(in_file, 'r', encoding='utf8') as f:
        data = [json.loads(line.strip()) for line in f]
    with open(src_file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(d['text'])
            f.write('\n')
    with open(dst_file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(d['abstract'])
            f.write('\n')

