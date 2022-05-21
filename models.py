import torch
from transformers import BertModel
import torch.nn as nn
from torch.autograd import Variable

class BertEM(nn.Module):
    def __init__(self, model, m, in_dim, device):
        super().__init__()
        self.model = BertModel.from_pretrained(model)
        with torch.cuda.device(device):
            self.nav = Variable(torch.randn(m, in_dim)).cuda()

    def forward(self, words):
        print (words)
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

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        # print ('size: ', (mask.size(1) - mask.float().sum(1)))
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)