import os
import json
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.optim import Adam
import importlib
import time

from config import NerConfig
# from models.LSNER import BertNer
from data_loader import NerDataset

from tqdm import tqdm
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification


class Trainer:
    def __init__(self,
                 output_dir=None,
                 model=None,
                 train_loader=None,
                 save_step=1000,
                 dev_loader=None,
                 test_loader=None,
                 optimizer=None,
                 schedule=None,
                 epochs=1,
                 device="cpu",
                 id2label=None):
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.schedule = schedule
        self.id2label = id2label
        self.save_step = save_step
        self.total_step = len(self.train_loader) * self.epochs

    def train(self):
        model_name = model_path.split(".")[-1]
        time_now = time.strftime("%Y%m%d-%H%M")
        global_step = 1
        loss_infos = {}
        mini_loss = float('inf')
        grad_steps = 0
        patience = 5000
        best_model = None
        for epoch in range(1, self.epochs + 1):
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
            for step, batch_data in loop:
                self.model.train()
                for key, value in batch_data.items():
                    batch_data[key] = value.to(self.device)
                input_ids = batch_data["input_ids"]
                # token_type_ids = batch_data['token_type_ids']
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]
                output = self.model(input_ids, attention_mask, labels)
                loss = output.loss
                loss_infos[global_step] = loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1)
                self.optimizer.step()
                self.schedule.step()
                loop.set_description(f'Epoch [{epoch}/{self.epochs}] {global_step}/{self.total_step}')
                loop.set_postfix(loss=loss.item(), mini_loss=mini_loss, grad_steps=grad_steps)
                global_step += 1
                # print(global_step)
                if global_step % self.save_step == 0:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.output_dir, f"{model_name}_{data_name}_{global_step}.bin"))
                # 如果当前损失值小于最小损失值
                if loss.item() < mini_loss:
                    mini_loss = loss.item()
                    best_model = self.model.state_dict()
                    grad_steps = 0
                else:
                    grad_steps += 1

                # 如果连续未下降步数超过阈值，终止训练
                if grad_steps >= patience:
                    print(
                        f"Loss has not improved for {patience} steps, stopping training.  mini_loss={mini_loss} steps={global_step}.")
                    plt.plot(loss_infos.keys(), loss_infos.values(), marker='')
                    # 添加标题和轴标签
                    plt.title('Loss over Steps')
                    plt.xlabel('Steps')
                    plt.ylabel('Loss')
                    # 保存图表为图片
                    plt.savefig(os.path.join(self.output_dir, "loss_over_steps.png"))

                    # 显示图表
                    plt.show()
                    torch.save(best_model, os.path.join(self.output_dir, f"{model_name}_{data_name}_best.bin"))
                    return  # 退出训练函数
            print(
                f'Epoch [{epoch}/{self.epochs}] {global_step}/{self.total_step} loss:{loss_infos[global_step - 1]} mini_loss={mini_loss} steps={grad_steps}')

        plt.plot(loss_infos.keys(), loss_infos.values(), marker='')
        # 添加标题和轴标签
        plt.title('Loss over Steps')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        # 保存图表为图片
        plt.savefig(os.path.join(self.output_dir, "loss_over_steps.png"))
        # 显示图表
        plt.show()

        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"{model_name}_{data_name}_last.bin"))

    def test(self):
        model_name = model_path.split(".")[-1]
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, f"{model_name}_{data_name}_best.bin")))
        self.model.eval()
        preds = []
        trues = []
        for step, batch_data in enumerate(tqdm(self.test_loader)):
            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
            input_ids = batch_data["input_ids"]
            # token_type_ids = batch_data['token_type_ids']
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]
            output = self.model(input_ids, attention_mask, labels)
            logits = output.logits
            attention_mask = attention_mask.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = sum(attention_mask[i])
                logit = logits[i][1:length]
                logit = [self.id2label[i] for i in logit]
                label = labels[i][1:length]
                label = [self.id2label[i] for i in label]
                preds.append(logit)
                trues.append(label)
                # print(logit)
                # print(label)
        report = classification_report(trues, preds)
        with open(os.path.join(self.output_dir, "classification_report.txt"), 'w', encoding='utf-8') as file:
            file.write(report)

        return report


def build_optimizer_and_scheduler(args, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    # optimizer_grouped_parameters = [
    #     # bert other module
    #     {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
    #      "weight_decay": args.weight_decay, 'lr': args.bert_learning_rate},
    #     {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.0, 'lr': args.bert_learning_rate},
    #
    #     # 其他模块，差分学习率
    #     {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
    #      "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
    #     {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.0, 'lr': args.crf_learning_rate},
    # ]
    #
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate, eps=args.adam_epsilon)
    # LSBER
    params = [
        {'params': model.token_encoder.parameters(), 'lr': 8e-5},  # BERT层的学习率
        # {'params': model.label_encoder.parameters(), 'lr': 1e-5},  # BERT层的学习率
        {'params': model.gcn.parameters(), 'lr': 0.8},  # GCN层的学习率
        {'params': model.bilstm.parameters(), 'lr': 1e-4},  # BiLSTM层的学习率
        {'params': model.linear.parameters(), 'lr': 1e-3},  # 线性层的学习率
        {'params': model.crf.parameters(), 'lr': 1e-4},  # CRF层的学习率
    ]
    optimizer = AdamW(params, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def main(data_name, model_path):
    args = NerConfig(data_name, model_path)

    with open(os.path.join(args.output_dir, "ner_args.json"), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.data_path, "train_zh.txt"), "r", encoding="utf-8") as fp:
        # train_data = fp.read().split("\n")
        train_data = NerDataset.process_file_CCKS(fp)
    # train_data = [json.loads(d) for d in train_data]

    with open(os.path.join(args.data_path, "dev_zh.txt"), "r", encoding="utf-8") as fp:
        dev_data = NerDataset.process_file_CCKS(fp)
        # dev_data = fp.read().split("\n")
        # dev_data = [json.loads(d) for d in dev_data]

    train_dataset = NerDataset(train_data, args, tokenizer)
    dev_dataset = NerDataset(dev_data, args, tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=2)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=2)

    NER_module = importlib.import_module(model_path)
    model = NER_module.BertNer(args)

    # for name,_ in model.named_parameters():
    #   print(name)

    model.to(device)
    t_toal = len(train_loader) * args.epochs
    optimizer, schedule = build_optimizer_and_scheduler(args, model, t_toal)

    train = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=dev_loader,
        optimizer=optimizer,
        schedule=schedule,
        epochs=args.epochs,
        device=device,
        id2label=args.id2label,
        save_step=args.save_step
    )

    train.train()

    report = train.test()
    print(report)


if __name__ == "__main__":
    data_name = "CCKS2019"
    model_path = "models.GLSNER"

    main(data_name, model_path)
