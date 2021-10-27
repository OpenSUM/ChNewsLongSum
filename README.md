# abstractiveSum


===
### Baseline model

见paper

### Our model
1. 预训练的Transformer(PDT+greedy/beam search)
2. PDT(前6层Transformer，后6层Transformer，非预训练的Transformer)




===========
### Dataset：LCSTS
备注：Original paper和ConvS2S和Global-encoding 采用的是the first part of LCSTS dataset for training, which contains 2.4M text-summary pairs, and 8K for validation and 725 pairs from the last part with high annotation scores as our test set.



### 实验结论可视化
1. input sentence length ROUGE-1,2,L
2. 中文数据集对于PDT(前6层Transformer，后6层Transformer，非预训练的Transformer)的可视化
