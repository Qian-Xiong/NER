import json

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
        token_embeddings = outputs.last_hidden_state
        token_embeddings = self.dropout(token_embeddings)
        tag_lens, hidden_size = self.label_representation.shape
        current_batch_size = token_embeddings.shape[0]
        label_embedding = self.label_representation.expand(current_batch_size, tag_lens, hidden_size)
        label_embeddings = label_embedding.transpose(2, 1)
        seq_out = torch.matmul(token_embeddings, label_embeddings)
        #seq_out = self.linear(seq_out)  # [batchsize, max_len, num_labels]

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
        logits=torch.argmax(logits, dim=-1).tolist()

        model_output = ModelOutput(logits, labels, loss)
        return model_output
