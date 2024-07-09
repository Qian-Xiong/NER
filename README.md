## 中文NER

### 环境
> python=3.11 \
> torch=2.3.0

### 数据集

数据集使用CCKS2017和CCKS2019，已经进行了预处理

### BERT模型

下载Bert预训练模型到pretrain_model/目录下

> huggingface-cli download --resume-download hfl/chinese-roberta-wwm-ext --local-dir ./model_hub/chinese-roberta-wwm-ext/
