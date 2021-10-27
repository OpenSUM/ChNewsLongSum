import os
import argparse

import rouge

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str)
parser.add_argument('--epochs', type=int)


def main(ckpt_dir, epochs):
    print('Calcing Rouge Scores...')
    os.system('python word2char.py --ckpt={}'.format(os.path.join(ckpt_dir, 'summary')))
    root = os.path.join(ckpt_dir, 'summary', 'split')
    rouger = rouge.Rouge()

    with open(os.path.join(root, 'ref_test'), 'r', encoding='utf8') as f:
        ref_test = [l.strip() for l in f]
    # with open(os.path.join(root, 'ref_eval'), 'r', encoding='utf8') as f:
    #     ref_eval = [l.strip() for l in f]

    with open(os.path.join(ckpt_dir, 'rouge.test'), 'w', encoding='utf8') as ftest, \
            open(os.path.join(ckpt_dir, 'rouge.eval'), 'w', encoding='utf8') as feval:
        for i in range(epochs):
            test_file = os.path.join(root, 'cand_test_best-%d' % i)
            print('Calc Rouge:', test_file)
            if os.path.exists(test_file):
                with open(test_file, 'r', encoding='utf8') as f:
                    cands = [l.strip() for l in f]
                scores = rouger.get_scores(cands, ref_test, avg=True)
                ftest.write('{}:\tRouge-1:{}\tRouge-2:{}\tRouge-L:{}\n'.format(
                    i, scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']))
                ftest.flush()
            continue
            eval_file = os.path.join(root, 'cand_eval_best-%d' % i)
            print('Calc Rouge:', eval_file)
            if os.path.exists(eval_file):
                with open(eval_file, 'r', encoding='utf8') as f:
                    cands = [l.strip() for l in f]
                scores = rouger.get_scores(cands, ref_eval, avg=True)
                feval.write('{}:\tRouge-1:{}\tRouge-2:{}\tRouge-L:{}\n'.format(
                    i, scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']))
                feval.flush()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.ckpt_dir, args.epochs)
