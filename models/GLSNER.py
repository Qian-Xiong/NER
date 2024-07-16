import json
import math
from random import random

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
        self.label_ids = self.__read_file(args.data_dir+args.data_name)
        hidden_size = self.bert_config.hidden_size
        self.lstm_hiden = 128
        self.max_seq_len = args.max_seq_len
        self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
                              dropout=0.1)
        self.linear = nn.Linear(self.lstm_hiden * 2, hidden_size)
        self.embedding = torch.nn.Embedding(self.bert_config.vocab_size, hidden_size)
        self.gcn = GCN(hidden_size, hidden_size)

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        self.batch_size = args.train_batch_size
        self.tag2id = args.label2id
        self.loss_f = CrossEntropyLoss()
        self.dropout = nn.Dropout(p=0.1)
        self.max_seq_len = args.max_seq_len
        self.sememe_emb = "att"

    def __read_file(self, file):
        with open(f"{file}/label_sememes_tree.json", "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        return data

    def forward(self, input_ids, attention_mask, labels=None):
        sememe_s = []
        for label_words in self.label_ids:
            span = []
            for word_list in label_words:
                if not word_list:
                    continue
                s = self.embedding(torch.tensor(word_list[0]).cuda().unsqueeze(0)).squeeze(0)
                senses_id_list = word_list[1]

                if len(senses_id_list) != 0:
                    sememe_tensor = []
                    for sense in senses_id_list:
                        nodes = sense[0]
                        adj = torch.FloatTensor(sense[1]).cuda()
                        nodes_tensor = torch.cat([torch.mean(
                            self.embedding(torch.tensor(node_tokens).cuda().unsqueeze(0)).squeeze(0),
                            dim=0).unsqueeze(0) for node_tokens in nodes], dim=0)
                        assert adj.shape[0] == nodes_tensor.shape[0]
                        out = self.gcn(nodes_tensor, adj)[0, :]
                        sememe_tensor.append(out)
                    sememe_tensor = torch.stack(sememe_tensor, dim=0)
                    if self.sememe_emb == "att":
                        distance = F.pairwise_distance(s, sememe_tensor, p=2)
                        attentionSocre = torch.softmax(distance, dim=0)
                        attentionSememeTensor = torch.einsum("a,ab->ab", attentionSocre, sememe_tensor)
                        span.append(torch.cat([attentionSememeTensor.mean(0).unsqueeze(0), s], dim=0).mean(0))
                    elif self.sememe_emb == "knn":
                        distance = F.pairwise_distance(s, sememe_tensor, p=2)
                        span.append(torch.cat([torch.stack(
                            [sememe_tensor[idx] for idx in torch.sort(distance, descending=True)[1][:3]],
                            dim=0).mean(0).unsqueeze(0), s], dim=0).mean(0))
                    else:
                        span.append(torch.cat([torch.stack(
                            [sememe_tensor[random.randint(0, sememe_tensor.shape[0])] for idx in range(3)],
                            dim=0).mean(0).unsqueeze(0), s], dim=0).mean(0))
            if len(span) != 0:
                sememe_s.append(torch.stack(span, dim=0).mean(0))
            else:
                sememe_s.append(torch.ones(self.bert_config.hidden_size).cuda())
        label_representation = torch.stack(sememe_s, dim=0)
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


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass=0, dropout=0.1):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):  # x特征矩阵,agj邻接矩阵
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # def forward(self, label_id):
    #     sememe_s = []
    #     for label_words in label_id:
    #         span = []
    #         for word_list in label_words:
    #             if not word_list:
    #                 continue
    #             s = self.embedding(torch.tensor(word_list[0]).cuda().unsqueeze(0)).squeeze(0)
    #             senses_id_list = word_list[1]
    #             if len(senses_id_list) != 0:
    #                 sememe_tensor = []
    #                 for sense in senses_id_list:
    #                     nodes = sense[0]
    #                     adj = torch.FloatTensor(sense[1]).cuda()
    #                     nodes_tensor = torch.cat([torch.mean(
    #                         self.embedding(torch.tensor(node_tokens).cuda().unsqueeze(0)).squeeze(0),
    #                         dim=0).unsqueeze(0) for node_tokens in nodes], dim=0)
    #                 assert adj.shape[0] == nodes_tensor.shape[0]
    #                 out = self.gcn(nodes_tensor, adj)[0, :]
    #                 sememe_tensor.append(out)
    #             sememe_tensor = torch.stack(sememe_tensor, dim=0)
    #         if self.sememe_emb == "att":
    #             distance = F.pairwise_distance(s, sememe_tensor, p=2)
    #             attentionSocre = torch.softmax(distance, dim=0)
    #             attentionSememeTensor = torch.einsum("a,ab->ab", attentionSocre, sememe_tensor)
    #             span.append(torch.cat([attentionSememeTensor.mean(0).unsqueeze(0), s], dim=0).mean(0))
    #         elif self.sememe_emb == "knn":
    #             distance = F.pairwise_distance(s, sememe_tensor, p=2)
    #             span.append(torch.cat([torch.stack(
    #                 [sememe_tensor[idx] for idx in torch.sort(distance, descending=True)[1][:3]],
    #                 dim=0).mean(0).unsqueeze(0), s], dim=0).mean(0))
    #         else:
    #             span.append(torch.cat([torch.stack(
    #                 [sememe_tensor[random.randint(0, sememe_tensor.shape[0])] for idx in range(3)],
    #                 dim=0).mean(0).unsqueeze(0), s], dim=0).mean(0))
    #         if len(span) != 0:
    #             sememe_s.append(torch.stack(span, dim=0).mean(0))
    #         else:
    #             sememe_s.append(torch.ones(self.hidden_size).cuda())
    #
    #     return torch.stack(sememe_s, dim=0)
