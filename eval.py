from transformers import BertTokenizer
from dataloader import *
from trainer import BERTtrainer
import torch
import numpy as np

import argparse

def load_config(filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    return dump['config']

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--device', type=int, default=0, help='gpu device to use.')
args = parser.parse_args()
opt = vars(args)

# load opt
model_file = args.model_dir + '/best_model.pt'
print("Loading model from {}".format(model_file))
opt = load_config(model_file)
opt['device'] = args.device
trainer = BERTtrainer(opt)
trainer.load(model_file)

tokenizer = BertTokenizer.from_pretrained(opt['bert'])

data_set = EpisodeDataset(opt['data_dir']+f'{args.dataset}', tokenizer)
data_batches = DataLoader(data_set, batch_size=1, collate_fn=collate_batch)

preds = []
golds = []
for db in data_batches:
    score, loss, labels = trainer.predict(db)
    preds += np.argmax(score.squeeze(2).data.cpu().numpy(), axis=1).tolist()
    golds += labels.cpu().tolist()

nrp = [0 if p >= 5 else 1 for p in preds]
nrg = [0 if g >= 5 else 1 for g in golds]

matched = [1 if p == nrg[i] and p == 1 else 0 for i, p in enumerate(nrp)]

print (sum(nrp), sum(nrg))

try:
    recall = sum(matched)/sum(nrg)
except ZeroDivisionError:
    recall = 0
try:
    precision = sum(matched)/sum(nrp)
except ZeroDivisionError:
    precision = 0
try:
    f1 = 2 * precision * recall / (precision + recall)
except ZeroDivisionError:
    f1 = 0

with open("NRC_output_%s.txt"%args.dataset.split('.')[0], 'w') as f:
    for i in range(0, len(nrp), data_set.query_size):
        output = ["no_relation" if p == 0 else "relation" for p in nrp[i:i+data_set.query_size]]
        f.write("\t".join(output))
        f.write('\n')

print ("current precision: %f, recall: %f, f1: %f"%(precision, recall, f1))

