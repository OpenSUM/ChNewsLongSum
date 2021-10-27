# 工作进度与安排

## 20190904

### 待完成工作：

- 多卡并行
    - [进行中]与沃老师协商Slurm集群分区限制
    - [已完成]多卡Demo
    - [已完成]多卡模型修改
        - 训练√
        - 测试√
    - [已完成]测试batch_size和训练速度：batch_size=48, CNN DAILY数据集250个Batch 332s，一天约10.8个Epoch
- 解码
    - [待决定]减小验证集(测试集)数量，可能留取10%数据用于平时测试，最终对比较好的ckpt用完整数据测一下最终结果
    - [待决定]提前结束预测（可能需要debug一下，排除错误）
    - [已完成]将Encoder和Decoder分开进行，先跑一遍Encoder得到输出，然后只跑Decoder依次得到完整摘要
    - [进行中]训练和测试分开进行，训练时仅生成ckpt不测试。
    - [已完成]多卡并行测试
- 模型
    - layer-aligned Transformer
    - bug fix: 从ckpt导入参数时，不排除optimization中的参数
