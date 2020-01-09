# 案情文本要素提取系统

## 数据说明

本项目采用的数据集源于[2019中国法研杯](http://cail.cipsc.org.cn/instruction.html)要素识别通道的部分数据集。

## 项目说明

见同一目录下的reprt.pdf。

## 项目效果

见同一目录下的show.mp4。

## 项目复现

本项目使用BERT微调预训练[哈工大讯飞联合实验室提出的预训练中文模型](https://github.com/ymcui/Chinese-BERT-wwm)。

```
@article{chinese-bert-wwm,
  title={Pre-Training with Whole Word Masking for Chinese BERT},
  author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing and Wang, Shijin and Hu, Guoping},
  journal={arXiv preprint arXiv:1906.08101},
  year={2019}
 }
```

