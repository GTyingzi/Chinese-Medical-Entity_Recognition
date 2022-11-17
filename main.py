# -*- coding: utf-8 -*-

import torch
from datetime import datetime
from torch.utils import data
import os
import warnings
import argparse
import numpy as np
from sklearn import metrics
from models import Bert_BiLSTM_CRF,Bert_CRF
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import NerDataset, PadBatch, VOCAB, tokenizer, tag2idx, idx2tag

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(e, model, iterator, optimizer, scheduler, device):
    model.train()
    losses = 0.0
    step = 0
    for i, batch in enumerate(iterator):
        step += 1
        x, y, z = batch
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        loss = model(x, y, z)
        losses += loss.item()
        """ Gradient Accumulation """
        '''
          full_loss = loss / 2                            # normalize loss 
          full_loss.backward()                            # backward and accumulate gradient
          if step % 2 == 0:             
              optimizer.step()                            # update optimizer
              scheduler.step()                            # update scheduler
              optimizer.zero_grad()                       # clear gradient
        '''
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print("Epoch: {}, Loss:{:.4f}".format(e, losses/step))

def validate(e, model, iterator, device,log_path):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            step += 1
            x, y, z = batch
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            y_hat = model(x, y, z, is_test=True)

            loss = model(x, y, z)
            losses += loss.item()
            # Save prediction
            for j in y_hat:
              Y_hat.extend(j)
            # Save labels
            mask = (z==1)
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig.cpu())

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = np.array(Y_hat)
    acc = (Y_hat == Y).mean()*100

    output = "{} Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(datetime.now(),e, losses/step, acc)
    print(output)
    with open(log_path,'a') as f: ## 将训练结果保存到日志中
        f.write(output + '\n')
        f.close()
    return model, losses/step, acc

def model_test(model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y, z = batch
            x = x.to(device)
            z = z.to(device)
            y_hat = model(x, y, z, is_test=True)
            # Save prediction
            for j in y_hat:
              Y_hat.extend(j)
            # Save labels
            mask = (z==1).cpu()
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig)

    Y = torch.cat(Y, dim=0).numpy()
    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]

    return y_true, y_pred

if __name__=="__main__":

    labels = ['B-BODY',
      'B-DISEASES',
      'B-DRUG',
      'B-EXAMINATIONS',
      'B-TEST',
      'B-TREATMENT',
      'I-BODY',
      'I-DISEASES',
      'I-DRUG',
      'I-EXAMINATIONS',
      'I-TEST',
      'I-TREATMENT']
    
    best_model = None
    _best_val_loss = 1e18
    _best_val_acc = 1e-18

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--trainset", type=str, default="./CCKS_2019_Task1/processed_data/train_dataset.txt")
    parser.add_argument("--validset", type=str, default="./CCKS_2019_Task1/processed_data/val_dataset.txt")
    parser.add_argument("--testset", type=str, default="./CCKS_2019_Task1/processed_data/test_dataset.txt")
    # parser.add_argument("--log",type=str,default="./logger/Bert_BiLSTM_CRF/train_acc.txt")
    parser.add_argument("--log", type=str, default="./logger/Bert_CRF/train_acc.txt")

    ner = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 选择模型
    # model = Bert_BiLSTM_CRF(tag2idx).cuda()
    model = Bert_CRF(tag2idx).cuda()
    print(f'Initial model:{model.model_name} Done.')

    # 加载数据集
    train_dataset = NerDataset(ner.trainset)
    eval_dataset = NerDataset(ner.validset)
    test_dataset = NerDataset(ner.testset)
    print('Load Data Done.')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=ner.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=PadBatch)

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=(ner.batch_size)//2,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=PadBatch)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(ner.batch_size)//2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch)

    #optimizer = optim.Adam(self.model.parameters(), lr=ner.lr, weight_decay=0.01)
    optimizer = AdamW(model.parameters(), lr=ner.lr, eps=1e-6)

    # Warmup
    len_dataset = len(train_dataset) 
    epoch = ner.n_epochs
    batch_size = ner.batch_size
    total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epoch
    
    warm_up_ratio = 0.1 # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)

    print('Start Train...,')
    for epoch in range(1, ner.n_epochs+1):

        train(epoch, model, train_iter, optimizer, scheduler, device) # 训练模型
        candidate_model, loss, acc = validate(epoch, model, eval_iter, device,ner.log) # 验证模型

        if loss < _best_val_loss and acc > _best_val_acc: # 将验证效果最好的模型保保留下来
          best_model = candidate_model
          _best_val_loss = loss
          _best_val_acc = acc

        print("=============================================")
    
    y_true, y_pred = model_test(best_model, test_iter, device) # 真实值和预测值
    output = metrics.classification_report(y_true, y_pred, labels=labels, digits = 3) # 计算真实值和预测值之间的相关指标

    print(output)
    with open(ner.log,'a') as f: ## 将训练结果保存到日志中
        f.write(output + '\n')
        f.close()


