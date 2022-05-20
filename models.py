import torch
from pytorch_pretrained_bert.modeling import BertModel

class BertEM(nn.Module):
    def __init__(self):
        super().__init__(model)
        self.model = BertModel.from_pretrained(model)

    def forward(self, inputs, subj_mask, obj_mask, pool_type):
        words = inputs
        h, pooled_output = self.model(words, output_all_encoded_layers=False)
        v = torch.cat([pool(h, subj_mask.eq(0), type=pool_type), pool(h, obj_mask.eq(0), type=pool_type)], 1)
        
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