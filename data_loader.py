import torch
import numpy as np

from torch.utils.data import Dataset


class NerDataset(Dataset):
    def __init__(self, data, args, tokenizer):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    @staticmethod
    def process_file_CCKS(data):
        datas = data.read().split("\n\n")
        data_sets = []
        for data in datas:
            data_set = {'text': [], "labels": []}
            lines = data.split("\n")
            for line in lines:
                if len(line.split("\t")) == 2:
                    data_set['text'].append(line.split("\t")[0])
                    data_set['labels'].append(line.split("\t")[1])
            data_sets.append(data_set)
        return data_sets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # #print("label2id",self.label2id)
        # text = self.data[item]["text"]
        # label = self.data[item]["labels"]
        # sentence = ''.join(text)
        # if len(text) >= self.max_seq_len - 2:
        #     label = label[0:self.max_seq_len - 2]
        # label = [self.label2id['O']] + [self.label2id[i] for i in label] + [self.label2id['O']]
        # if len(label) < self.max_seq_len:
        #     label = label + [self.label2id['O']] * (self.max_seq_len - len(label))
        #
        # assert len(label) == self.max_seq_len
        # # tags.append(label)
        #
        # inputs = self.tokenizer.encode_plus(sentence, max_length=self.max_seq_len, pad_to_max_length=True,
        #                                         return_tensors='pt')
        # input_ids, token_type_ids, attention_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
        #         'attention_mask']
        # # input_ids = torch.tensor(np.array(input_id)).long()
        # # token_type_ids = torch.tensor(np.array(token_type_ids)).long()
        # # attention_mask = torch.tensor(np.array(attention_mask)).long()
        # label = torch.tensor(np.array(label)).long()
        #
        #
        # data = {
        #     "input_ids": input_ids.squeeze(0),
        #     "attention_mask": attention_mask.squeeze(0),
        #     # "token_type_ids": token_type_ids.squeeze(0),
        #     "labels": label,
        # }

        text = self.data[item]["text"]
        labels = self.data[item]["labels"]
        if len(text) > self.max_seq_len - 2:
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        attention_mask = [1] * len(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))

        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data


