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

data_set = EpisodeDataset(opt['data_dir']+f'{args.dataset}_episode.json', tokenizer)
data_batches = DataLoader(data_set, batch_size=opt['batch_size'], collate_fn=collate_batch)

preds = []
for db in data_batches:
    score, loss = trainer.predict(db)
    preds += np.argmax(score.squeeze(2).data.cpu().numpy(), axis=1).tolist()
nrp = [p == 5 for p in preds]
nrg = [g == 5 for g in data_set.get_golds()]
acc = sum([nrp[i] == nrg[i] for i in range(len(nrp))])/len(nrp)
print ("current accuracy: %f"%acc)

