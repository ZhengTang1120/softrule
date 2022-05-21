"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import BertEM
from transformers import AdamW, BertConfig
from transformers.optimization import get_linear_schedule_with_warmup

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
            query = batch[0].cuda()
            support_sents = batch[1].cuda()
            labels = batch[2].cuda()
    else:
        query = batch[0]
        support_sents = batch[1]
        labels = batch[2]
    batch_size = query.size(0)
    N = support_sents.size(1)
    k = support_sents.size(2)
    return query, support_sents, labels, N, k, batch_size


class BERTtrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        config = BertConfig.from_pretrained(opt['bert'])
        self.in_dim = config.hidden_size * 2
        self.encoder = BertEM(opt['bert'], opt['m'], self.in_dim)
        with torch.cuda.device(opt['device']):
            self.nav = torch.rand((opt['m'], self.in_dim), requires_grad=True, device="cuda")
        self.criterion = nn.CrossEntropyLoss()

        param_optimizer = list(self.encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': self.nav, 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=opt['lr'])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
            num_warmup_steps=opt['num_warmup_steps'], 
            num_training_steps=opt['num_training_steps'])

        if opt['cuda']:
            with torch.cuda.device(opt['device']):
                self.encoder.cuda()
                self.criterion.cuda()

    def update(self, batch):
        query, support_sents, labels, N, k, batch_size = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        self.encoder.train()
        qv = self.encoder(query)
        print (qv)
        svs = self.encoder(support_sents.view(batch_size*N*k, -1))
        print (svs)
        svs = torch.mean(svs.view(batch_size, N, k, -1), 2)
        svs = torch.cat([svs, self.nav.expand(batch_size, -1,self.in_dim)], 1)
        loss = self.criterion(torch.bmm(svs, qv.view(batch_size, -1, 1)), labels.view(batch_size, 1))
        loss_val = loss.item()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        qv = svs = query = support_sents = None
        return loss_val

    def predict(self, batch):
        query, support_sents, labels, N, k, batch_size = unpack_batch(batch, self.opt['cuda'], self.opt['device'])
        self.encoder.eval()
        with torch.no_grad():
            qv = self.encoder(query)
            svs = self.encoder(support_sents.view(batch_size*N*k, -1))
            svs = torch.mean(svs.view(batch_size, N, k, -1), 2)
            svs = torch.cat([svs, self.nav.expand(batch_size, -1,self.in_dim)], 1)
            scores = torch.bmm(svs, qv.view(batch_size, -1, 1))
            loss = self.criterion(scores, labels.view(batch_size, 1)).item()
            qv = svs = query = support_sents = None
            return scores, loss

