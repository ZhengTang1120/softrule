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
        output = self.model(words)
        # v = torch.cat([pool(h, subj_mask.eq(0), type=pool_type), pool(h, obj_mask.eq(0), type=pool_type)], 1)
        return output.pooler_output

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