set -e
set -v

threshold=30

#cat train.src.token train.dst.token | subword-nmt learn-bpe -s 32000 -o train.codec
#cat train.src.token train.dst.token | subword-nmt apply-bpe -c train.codec > train.sub
#subword-nmt get-vocab < train.sub > vocab
subword-nmt apply-bpe -c train.codec --vocabulary vocab --vocabulary-threshold ${threshold} < train.src.token > train.src.bpe.token
subword-nmt apply-bpe -c train.codec --vocabulary vocab --vocabulary-threshold ${threshold} < train.dst.token > train.dst.bpe.token
subword-nmt apply-bpe -c train.codec --vocabulary vocab --vocabulary-threshold ${threshold} < test.src.token > test.src.bpe.token
subword-nmt apply-bpe -c train.codec --vocabulary vocab --vocabulary-threshold ${threshold} < test.dst.token > test.dst.bpe.token
subword-nmt apply-bpe -c train.codec --vocabulary vocab --vocabulary-threshold ${threshold} < eval.src.token > eval.src.bpe.token
subword-nmt apply-bpe -c train.codec --vocabulary vocab --vocabulary-threshold ${threshold} < eval.dst.token > eval.dst.bpe.token

