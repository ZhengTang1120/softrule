"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

from models import BertEM, MLP, pool
from transformers import AdamW, BertConfig
from transformers.optimization import get_linear_schedule_with_warmup

import random

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
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.mlp.load_state_dict(checkpoint['mlp'])

    def save(self, filename):
        params = {
                'encoder': self.encoder.state_dict(),
                'mlp': self.mlp.state_dict(),
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
            query = batch[0].cuda()
            labels = batch[1].cuda()
    else:
        query = batch[0]
        labels = batch[1]
    batch_size = query.size(0)
    return query, labels, batch_size


class BERTtrainer(Trainer):
    def __init__(self, opt, notas=None):
        self.opt = opt
        config = BertConfig.from_pretrained(opt['bert'])
        self.in_dim = config.hidden_size * 2
        self.hidden_dim = opt['hidden_dim']
        self.encoder = BertEM(opt['bert'])
        self.mlp = MLP(self.in_dim, opt['hidden_dim'])
        self.criterion = nn.BCELoss()

        param_optimizer = list(self.encoder.named_parameters())
        mlp_param = list(self.mlp.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-07},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in mlp_param], 'weight_decay': 1e-03}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=opt['lr'])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
            num_warmup_steps=opt['num_warmup_steps'], 
            num_training_steps=opt['num_training_steps'])
        if opt['cuda']:
            with torch.cuda.device(opt['device']):
                self.encoder.cuda()
                self.mlp.cuda()
                self.criterion.cuda()

    def update(self, batch):
        query, labels, batch_size = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        self.encoder.train()
        self.mlp.train()
        qv = self.encoder(query)
        scores = self.mlp(qv)
        loss = self.criterion(scores, labels.view(batch_size, 1).float())
        loss_val = loss.item()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        qv = svs = query = None
        return loss_val

    def predict(self, batch):
        query, labels, batch_size = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        self.encoder.eval()
        self.mlp.eval()
        with torch.no_grad():
            scores = self.mlp(self.encoder(query))
            loss = self.criterion(scores, labels.view(batch_size, 1).float()).item()
            qv = svs = query = None
            return scores, loss, labels
