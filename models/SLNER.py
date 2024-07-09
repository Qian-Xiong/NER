import json
import math

import torch
import torch.nn as nn

from torchcrf import CRF
from transformers import BertModel, BertConfig, BertTokenizer
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class ModelOutput:
    def __init__(self, logits, labels, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss


class BertNer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token_encoder = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir)
        hidden_size = self.bert_config.hidden_size
        self.lstm_hiden = 128
        self.max_seq_len = args.max_seq_len
        self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
                              dropout=0.1)
        self.linear = nn.Linear(self.lstm_hiden * 2, hidden_size)
        self.label_encoder = BertModel.from_pretrained(args.bert_dir)
        self.label_context = self.__read_file(args.tag_file)
        self.index_context = {
            "B": "开始词",
            "I": "中间词",
            "E": "结束词",
            "S": "单字词"
        }
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        # self.label_representation = self.build_label_representation(tag2id).to(self.device)
        self.batch_size = args.train_batch_size
        self.tag2id = args.label2id
        self.loss_f = CrossEntropyLoss()
        self.dropout = nn.Dropout(p=0.1)
        self.max_seq_len = args.max_seq_len
        # self.crf = CRF(args.num_labels, batch_first=True)
       # self.linear = nn.Linear(hidden_size, args.num_labels)


    def __read_file(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def build_label_representation(self, tag2id):
        labels = []
        for k, v in tag2id.items():
            if k.split('-')[-1] != 'O':
                idx, label = k.split('-')[0], k.split('-')[-1]
                label = self.label_context[label]
                labels.append(label + self.index_context[idx])
            else:
                labels.append("其他类别词")
        '''
        mutul(a,b) a和b维度是否一致的问题
        A.shape =（b,m,n)；B.shape = (b,n,k)
        torch.matmul(A,B) 结果shape为(b,m,k)
        '''

        tag_max_len = max([len(l) for l in labels])
        tag_embeddings = []
        for label in labels:
            input_ids = self.tokenizer.encode_plus(label, return_tensors='pt', padding='max_length',
                                                   max_length=tag_max_len)
            outputs = self.label_encoder(input_ids=input_ids['input_ids'].to(self.device),
                                         token_type_ids=input_ids['token_type_ids'].to(self.device),
                                         attention_mask=input_ids['attention_mask'].to(self.device))
            pooler_output = outputs.pooler_output
            pooler_output = self.dropout(pooler_output)
            tag_embeddings.append(pooler_output)
        label_embeddings = torch.stack(tag_embeddings, dim=0)
        label_embeddings = label_embeddings.squeeze(1)
        return label_embeddings

    def forward(self, input_ids, attention_mask, labels=None):

        label_representation = self.build_label_representation(self.tag2id)
        self.label_representation = label_representation.detach()
        outputs = self.token_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs[0]  # [batchsize, max_len, 768]
        batch_size = token_embeddings.size(0)
        seq_out, _ = self.bilstm(token_embeddings)
        seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
        seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
        token_embeddings = self.linear(seq_out)
        tag_lens, hidden_size = self.label_representation.shape
        current_batch_size = token_embeddings.shape[0]
        label_embedding = self.label_representation.expand(current_batch_size, tag_lens, hidden_size)
        label_embeddings = label_embedding.transpose(2, 1)
        seq_out = torch.matmul(token_embeddings, label_embeddings)
        # seq_out = self.linear(seq_out)  # [batchsize, max_len, num_labels]

        # 计算交叉熵损失
        loss = None
        if labels is not None:
            # Flatten the predictions and labels
            seq_out = seq_out.view(-1, seq_out.size(-1))  # [batch_size * max_len, num_labels]
            labels = labels.view(-1)  # [batch_size * max_len]
            loss = nn.CrossEntropyLoss()(seq_out, labels.long())

        # 使用softmax获取预测概率
        logits = nn.Softmax(dim=-1)(seq_out)

        # 将logits重新形状为[batch_size, max_len, num_labels]
        logits = logits.view(-1, self.max_seq_len, seq_out.size(-1))
        logits = torch.argmax(logits, dim=-1).tolist()

        model_output = ModelOutput(logits, labels, loss)
        return model_output


class Biaffine(nn.Module):
  def __init__(self, in1_features: int, in2_features: int, out_features: int):
    super().__init__()
    self.bilinear = PairwiseBilinear(in1_features + 1, in2_features + 1, out_features)
    self.bilinear.weight.data.zero_()
    self.bilinear.bias.data.zero_()

  def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], dim=input1.dim() - 1)
    input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], dim=input2.dim() - 1)
    return self.bilinear(input1, input2)


class PairwiseBilinear(nn.Module):
  """
  https://github.com/stanfordnlp/stanza/blob/v1.1.1/stanza/models/common/biaffine.py#L5  # noqa
  """

  def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True):
    super().__init__()
    self.in1_features = in1_features
    self.in2_features = in2_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.Tensor(in1_features, out_features, in2_features))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter("bias", None)
    self.reset_parameters()

  def reset_parameters(self):
    bound = 1 / math.sqrt(self.weight.size(0))
    nn.init.uniform_(self.weight, -bound, bound)
    if self.bias is not None:
      nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    d1, d2, out = self.in1_features, self.in2_features, self.out_features
    n1, n2 = input1.size(1), input2.size(1)
    # (b * n1, d1) @ (d1, out * d2) => (b * n1, out * d2)
    x1W = torch.mm(input1.view(-1, d1), self.weight.view(d1, out * d2))
    # (b, n1 * out, d2) @ (b, d2, n2) => (b, n1 * out, n2)
    x1Wx2 = x1W.view(-1, n1 * out, d2).bmm(input2.transpose(1, 2))
    y = x1Wx2.view(-1, n1, self.out_features, n2).transpose(2, 3)
    if self.bias is not None:
      y.add_(self.bias)
    return y  # (b, n1, n2, out)

  def extra_repr(self) -> str:
    return "in1_features={}, in2_features={}, out_features={}, bias={}".format(
      self.in1_features, self.in2_features, self.out_features, self.bias is not None
    )