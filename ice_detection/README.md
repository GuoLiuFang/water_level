# 冰凌显著性检测模型训练文档

## 1. 环境版本要求

python==3.6

torch==1.3.1

numpy==1.17.1

## 2. 模型训练

在train.py里面进行相应路径的配置：

1. image_root: 原始图片的路径
2. gt_root: 掩码图片的路径
3. save_path: 模型保存的路径

主要的配置为以上三项，其他诸如"epoch", "lr", "batchsize"等参数可通过命令行传参进行修改

当完成以上配置文件修改之后，使用以下命令即可开启模型的训练:

```python
python train.py --epoch 100 --lr 1e-4 --batchsize 128 --isResNet True
```

## 3. 模型预测

在predict_cpd.py的主函数里面进行相应路径的配置：

1. test_size: 图片进行resize的尺寸
2. model_path: 模型加载的路径
3. image_path: 原始图片的路径
4. save_path: 输出显著性检测之后的图片路径

当完成以上配置文件修改之后，使用以下命令即可开启模型的训练:

```python
python predict_cpd.py
```

举例而言，原图和预测结果分为如下：

| 原图                                                         | 预测结果                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="/Users/yangbitao/Library/Application Support/typora-user-images/image-20210114163925858.png" alt="image-20210114163925858" style="zoom:33%;" /> | <img src="/Users/yangbitao/Library/Application Support/typora-user-images/image-20210114164004722.png" alt="image-20210114164004722" style="zoom:33%;" /> |

## 4. 模型测试指标

以F1值作为测试指标，该模型在各个开源数据集的测试情况如下：

| Model | FPS  | ECSSD | HKU-IS | DUT-OMRON | DUTS-TEST | PASCAL-S |
| ----- | ---- | ----- | ------ | --------- | --------- | -------- |
| CPD-R | 62   | 0.939 | 0.925  | 0.797     | 0.865     | 0.864    |

以mean average error作为测试指标，该模型在各个开源数据集的测试情况如下：

| Model | ECSSD | HKU-IS | DUT-OMRON | DUTS-TEST | PASCAL-S |
| ----- | ----- | ------ | --------- | --------- | -------- |
| CPD-R | 0.037 | 0.034  | 0.056     | 0.043     | 0.072    |

