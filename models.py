import torch
from transformers import BertModel
import torch.nn as nn
from torch.autograd import Variable

class BertEM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = BertModel.from_pretrained(model)

    def forward(self, words):
        output = self.model(words)
        h = output.last_hidden_state
        subj_mask = torch.logical_and(words.unsqueeze(2).gt(0), words.unsqueeze(2).lt(3))
        obj_mask = torch.logical_and(words.unsqueeze(2).gt(2), words.unsqueeze(2).lt(20))
         for i, x in enumerate(torch.sum(subj_mask, 1)):
            if x[0].item() == 0:
                print ("subj missing", words[i])
        for i, x in enumerate(torch.sum(obj_mask, 1)):
            if x[0].item() == 0:
                print ("obj missing", words[i])
        v = torch.cat([pool(h, subj_mask.eq(0), type='avg'), pool(h, obj_mask.eq(0), type='avg')], 1)
        return v

def pool(h, mask=None, type='max'):
    if type == 'max':
        if mask:
            h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        if mask is not None:
            h = h.masked_fill(mask, 0)
            return h.sum(1) / (mask.size(1) - mask.float().sum(1))
        else:
            return h.sum(1) / h.size(1)
    else:
        if mask:
            h = h.masked_fill(mask, 0)
        return h.sum(1)