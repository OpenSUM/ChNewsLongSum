import os
import sys
import re
import json
from multiprocessing import Pool

from pyltp import Segmentor

LTP_DIR = r'/home/LAB/tangb/ltp_data_v3.4.0'
print('Loading Segmentor Model...')
segmentor = Segmentor()
segmentor.load(os.path.join(LTP_DIR, 'cws.model'))


def segment_text(text):
    text = re.sub('\n+', '\n', text)
    sentences = text.split('\n')
    result = []
    for s in sentences:
        shorts = s.split(' ')
        for short in shorts:
            result.extend(list(segmentor.segment(short)))
        result.append('<ln>')
    return '<cls> ' + ' '.join(result).strip('<ln>').strip() + ' <sep>'


def _segment(line):
    js = json.loads(line.strip())
    js['abstract'] = segment_text(js['abstract'])
    js['title'] = segment_text(js['title'])
    js['text'] = segment_text(js['text'])
    return js


if __name__ == '__main__':
    print('Loading Data File...')
    with open(sys.argv[1], 'r', encoding='utf8') as f:
        data = f.readlines()
    print('Segmenting Texts...')
    pool = Pool(25)
    with open(sys.argv[2], 'w', encoding='utf8') as f:
        offset = 1
        span = 5000
        while (offset-1) * span < len(data):
            result = pool.map(_segment, data[(offset-1)*span:offset*span])
            print('%d samples segmented...' % (offset * span))
            for i, t in enumerate(result):
                f.write(json.dumps(t, ensure_ascii=False))
                f.write('\n')
            offset += 1
        print('All samples have been segmented!')
