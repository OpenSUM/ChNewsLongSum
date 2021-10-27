from collections import Counter

ct = Counter()
for filename in ['./train.src', './train.dst']:
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            ct.update(line.strip().split(' '))
with open('vocab.txt', 'w', encoding='utf8') as f:
    for w, c in ct.most_common():
        f.write('{} {}\n'.format(w, c))

