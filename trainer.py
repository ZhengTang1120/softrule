"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from models import BertEM
from pytorch_pretrained_bert.optimization import BertAdam

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.tagger.load_state_dict(checkpoint['tagger'])
        device = self.opt['device']
        self.opt = checkpoint['config']
        self.opt['device'] = device

    def save(self, filename):
        params = {
                'classifier': self.classifier.state_dict(),
                'encoder': self.encoder.state_dict(),
                'tagger': self.tagger.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

def unpack_batch(batch, cuda=False, device=0):
    if cuda:
        with torch.cuda.device(device):
            query = ?
            support_sents = ?
    else:
        query = ?
        support_sents = ?
    return query, support_sents


class BERTtrainer(Trainer):
    def __init__(self, opt):
        self.in_dim = 2048
        self.encoder = BertEM("spanbert-large-cased")
        self.criterion = nn.CrossEntropyLoss()
        self.nav = Variable(torch.randn(opt['m'], self.in_dim))

        param_optimizer = list(self.encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = BertAdam(optimizer_grouped_parameters,
             lr=opt['lr'],
             warmup=opt['warmup_prop'],
             t_total= opt['train_batch'] * self.opt['num_epoch'])

    def update(self, batch):
        query, support_sents = unpack_batch(batch)
        self.encoder.train()
        qv = self.encoder(query)
        svs = self.encoder(support_sents.reshape(batch_size, N*k, -1))
        svs = torch.mean(svs.reshape(batch_size, N, k, -1), 2)
        svs = torch.cat([svs, self.nav.expand(batch_size, -1,self.in_dim)], 1)

        loss = self.criterion(torch.bmm(svs, qv.view(batch_size, -1, 1)))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def predict(self, batch):
        query, support_sents = unpack_batch(batch)
        self.encoder.eval()
        qv = self.encoder(query)
        svs = self.encoder(support_sents.reshape(batch_size, N*k, -1))
        svs = torch.mean(svs.reshape(batch_size, N, k, -1), 2)
        svs = torch.cat([svs, self.nav.expand(batch_size, -1,self.in_dim)], 1)
        loss = self.criterion(torch.bmm(svs, qv.view(batch_size, -1, 1)))

        scores = torch.bmm(svs, qv.view(batch_size, -1, 1))
        return scores, loss.item()

