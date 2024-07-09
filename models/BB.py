import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from torchcrf import CRF
from transformers import BertModel, BertConfig
import torch.nn.functional as F


class ModelOutput:
    def __init__(self, logits, labels, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss


class BertNer(nn.Module):
    def __init__(self, args):
        super(BertNer, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir)
        hidden_size = self.bert_config.hidden_size
        self.max_seq_len = args.max_seq_len

        # 直接从BERT的输出映射到标签
        self.linear = nn.Linear(hidden_size, args.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_output[0]  # [batchsize, max_len, hidden_size]
        seq_out = self.linear(seq_out)  # [batchsize, max_len, num_labels]

        # 计算交叉熵损失
        loss = None
        if labels is not None:
            # Flatten the predictions and labels
            seq_out = seq_out.view(-1, seq_out.size(-1))  # [batch_size * max_len, num_labels]
            labels = labels.view(-1)  # [batch_size * max_len]
            loss = nn.CrossEntropyLoss()(seq_out, labels.long())

        # 使用softmax获取预测概率
        logits = F.softmax(seq_out, dim=-1)

        # 将logits重新形状为[batch_size, max_len, num_labels]
        logits = logits.view(-1, self.max_seq_len, seq_out.size(-1))

        logits = torch.argmax(logits, dim=-1).tolist()

        model_output = ModelOutput(logits, labels, loss)
        return model_output
