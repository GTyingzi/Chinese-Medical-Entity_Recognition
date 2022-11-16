# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertModel
from TorchCRF import CRF

class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(self.tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        #self.crf = CRF(self.tagset_size, batch_first=True)
        self.crf = CRF(self.tagset_size)
    
    def _get_features(self, sentence): # sentence：{batch_size,seq_Len}
        with torch.no_grad():
            encoder_output = self.bert(sentence)
            embeds = encoder_output[0] # embeds：{batch_size,seq_len,embedding_dim}
        enc, _ = self.lstm(embeds) # enc：{batch_size,seq_len,hidden_dim}
        enc = self.dropout(enc)
        feats = self.linear(enc) # feats：{batch_size,seq_len,target_size}
        return feats

    def forward(self, sentence, tags, mask, is_test=False): # {batch_size,seq_Len}
        emissions = self._get_features(sentence) # 得到特征分数,emissions：{batch_size,seq_len,target_size}
        if not is_test: # Training，return loss
            loss=-self.crf.forward(emissions, tags, mask).mean()
            return loss
        else: # Testing，return decoding
            decode=self.crf.viterbi_decode(emissions, mask)
            return decode
 
