import re
import time

import numpy as np

from constants import SAMPLE_LIMIT
from constants import SEP_TOKEN, UNK_TOKEN, CLS_TOKEN
from . import tokenization


def load_vocab(vocab_file, do_lower=False):
    word2id, id2word = dict(), dict()
    with open(vocab_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            w = line.strip('\n')
            if do_lower:
                if len(w) < 3 or not w[0] == '[' or not w[-1] == ']':
                    w = w.lower()
            word2id[w] = i
            id2word[i] = w
    assert len(word2id) == len(id2word)
    for i in range(len(word2id)):
        assert word2id[id2word[i]] == i
    return word2id, id2word


def load_text(file, do_lower, vocab, substr_prefix='##'):
    print('Loading file: {}'.format(file))

    tokenizer = tokenization.FullTokenizer(vocab=vocab, do_lower_case=do_lower, substr_prefix=substr_prefix)
    i = -1
    with open(file, 'r', encoding='utf8') as f:
        tokens = []
        for i, l in enumerate(f):
            l = tokenizer.tokenize(l)
            tokens.append(l)

            if i % 100 == 0:
                print('\r{}'.format(i), end='')
    print('\r{}/{}'.format(i + 1, i + 1))
    return tokens


def load_src(src_file, seq_length, do_lower, vocab=None, substr_prefix='##', blank_reg=re.compile('[ \t\n]'),
             limit=SAMPLE_LIMIT):
    print('Loading src file: {}'.format(src_file))

    if src_file.endswith('.token'):
        tokenize = lambda x: x.strip().lstrip('<cls>').rstrip('<sep>').strip().split(' ')
    else:
        tokenizer = tokenization.FullTokenizer(vocab=vocab, do_lower_case=do_lower, substr_prefix=substr_prefix)
        tokenize = lambda x: tokenizer.tokenize(x)

    with open(src_file, 'r', encoding='utf8') as f:
        #     tokens = [list(blank_reg.sub('', line)) for line in f]
        tokens = []
        mask = []
        ids = []
        ids_extend = []
        oovs = []
        oov_size = []
        for i, l in enumerate(f):
            l = tokenize(l)
            if vocab:
                oov = []
                tmp_token = [CLS_TOKEN]
                tmp_extend = [vocab[CLS_TOKEN]]
                tmp = [vocab[CLS_TOKEN]]
                for w in l[:seq_length - 2]:
                    tmp_token.append(w)
                    if w in vocab:
                        tmp.append(vocab[w])
                        tmp_extend.append(vocab[w])
                    elif w in oov:
                        tmp.append(vocab[UNK_TOKEN])
                        tmp_extend.append(len(vocab) + oov.index(w))
                    else:
                        oov.append(w)
                        tmp.append(vocab[UNK_TOKEN])
                        tmp_extend.append(len(vocab) + oov.index(w))
                tmp_token.append(SEP_TOKEN)
                tmp_extend.append(vocab[SEP_TOKEN])
                tmp.append(vocab[SEP_TOKEN])
                mask.append(([1] * len(tmp_extend) + [0] * (seq_length - len(tmp_extend)))[:seq_length])

                tmp_extend += [0] * (seq_length - len(tmp_extend))
                tmp += [0] * (seq_length - len(tmp))

                tokens.append(tmp_token)
                ids_extend.append(tmp_extend[:seq_length])
                ids.append(tmp[:seq_length])
                oovs.append(oov)
                oov_size.append(len(oov))
            else:
                mask.append(([1] * len(l) + [0] * (seq_length - len(l)))[:seq_length])

            if i % 1000 == 0:
                print('\r{}/{}'.format(i, limit), end='')
            if limit is not None and len(ids) >= limit:
                break
    print('\r{}/{}'.format(i + 1, i + 1))
    if vocab:
        return np.array(tokens), np.array(ids), np.array(ids_extend), np.array(mask), np.array(oov_size), np.array(oovs)
    else:
        return np.array(tokens), np.array(mask)


def load_dst(dst_file, seq_length, do_lower, vocab=None, src_oovs=None, src_ids=None, substr_prefix='##',
             blank_reg=re.compile('[ \t\n]'),
             limit=SAMPLE_LIMIT):
    print('Loading dst file: {}'.format(dst_file))
    if src_oovs is not None or src_ids is not None:
        assert vocab is not None
        assert src_oovs is not None
        assert src_ids is not None
        assert len(src_oovs) == len(src_ids)

    if vocab:
        vocab_size = len(vocab)

    if dst_file.endswith('.token'):
        tokenize = lambda x: x.strip().lstrip('<cls>').rstrip('<sep>').strip().split(' ')
    else:
        tokenizer = tokenization.FullTokenizer(vocab=vocab, do_lower_case=do_lower, substr_prefix=substr_prefix)
        tokenize = lambda x: tokenizer.tokenize(x)

    with open(dst_file, 'r', encoding='utf8') as f:
        tokens = []
        mask = []
        ids = []
        ids_extend = []
        ids_ext_sep = []
        for i, l in enumerate(f):
            l = tokenize(l)
            tokens.append(l)
            if vocab:
                tmp_extend = []
                tmp = []
                for w in l[:seq_length - 1]:
                    if w != UNK_TOKEN and w in vocab:
                        tmp_extend.append(vocab[w])
                    elif w in src_oovs[i]:
                        tmp_extend.append(vocab_size + src_oovs[i].index(w))
                    else:
                        tmp_extend.append(vocab[UNK_TOKEN])
                    tmp.append(vocab[w] if w in vocab else vocab[UNK_TOKEN])
                mask.append(([1] * (len(tmp_extend) + 1) + [0] * (seq_length - len(tmp_extend) - 1))[:seq_length])
                ids_ext_sep.append(tmp_extend + [vocab[SEP_TOKEN]] + [0] * (seq_length - len(tmp_extend) - 1))

                tmp_extend += [0] * (seq_length - len(tmp_extend))
                tmp += [0] * (seq_length - len(tmp))

                ids_extend.append(tmp_extend[:seq_length])
                ids.append(tmp[:seq_length])
            else:
                mask.append(([1] * len(l) + [0] * (seq_length - len(l)))[:seq_length])

            if i % 1000 == 0:
                print('\r{}/{}'.format(i, limit), end='')
            if limit is not None and len(ids) >= limit:
                break
    print('\r{}/{}'.format(i + 1, i + 1))
    if vocab:
        return np.array(tokens[:limit]), np.array(ids), np.array(ids_extend), np.array(ids_ext_sep), np.array(mask)
    else:
        return np.array(tokens[:limit]), np.array(mask)


def count(src_file, dst_file, vocab, substr_prefix='##'):
    start_time = time.time()
    tokenizer = tokenization.FullTokenizer(vocab=vocab, do_lower_case=True, substr_prefix=substr_prefix)
    print('counting src...')
    with open(src_file, 'r', encoding='utf8') as f:
        #     tokens = [list(blank_reg.sub('', line)) for line in f]
        src_lengths = []
        for i, l in enumerate(f):
            l = tokenizer.tokenize(l)
            src_lengths.append(len(l))
            if i % 2000 == 0:
                print('\r{}, time: {:.2f}s'.format(i, time.time() - start_time), end='')
        print('\r{}, time: {:.2f}s'.format(i + 1, time.time() - start_time))
    src_lengths.sort()
    print('\ncounting dst...')
    with open(dst_file, 'r', encoding='utf8') as f:
        dst_lengths = []
        for i, l in enumerate(f):
            l = tokenizer.tokenize(l)
            dst_lengths.append(len(l))
            if i % 2000 == 0:
                print('\r{}, time: {:.2f}s'.format(i, time.time() - start_time), end='')
        print('\r{}, time: {:.2f}s'.format(i + 1, time.time() - start_time))
    dst_lengths.sort()
    print('Time of Loading Training Data: {:.2f}s\n'.format(time.time() - start_time))
    print('\nSRC:\nMax:{max}\nMin:{min}\nAverage:{avg}\n95%:{r95}\n98%:{r98}'.format(
        max=max(src_lengths),
        min=min(src_lengths),
        avg=sum(src_lengths) / len(src_lengths),
        r95=src_lengths[round(len(src_lengths) * 0.95)],
        r98=src_lengths[round(len(src_lengths) * 0.98)],
    ))
    print()
    print('DST:\nMax:{max}\nMin:{min}\nAverage:{avg}\n95%:{r95}\n98%:{r98}'.format(
        max=max(dst_lengths),
        min=min(dst_lengths),
        avg=sum(dst_lengths) / len(dst_lengths),
        r95=dst_lengths[round(len(dst_lengths) * 0.95)],
        r98=dst_lengths[round(len(dst_lengths) * 0.98)],
    ))
    return src_lengths, dst_lengths


def id2text(ids, id2word, oov, vocab_size=None):
    if vocab_size is None:
        vocab_size = len(id2word)
    text = []
    if type(oov) != list:
        oov = oov.tolist()
    for i in ids:
        if i in id2word:
            text.append(id2word[i])
        else:
            text.append(oov[i - vocab_size])
    return ' '.join(text)


def ids2text(ids, id2word, oovs):
    vocab_size = len(id2word)
    texts = []
    for id_, oov in zip(ids, oovs):
        texts.append(id2text(ids=id_, id2word=id2word, oov=oov, vocab_size=vocab_size))
    return texts
