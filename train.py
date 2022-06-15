from transformers import BertTokenizer
from dataloader import *
from trainer import BERTtrainer
import numpy as np

import argparse
import os

from torch.utils.data import RandomSampler
import torch
import random

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./')
parser.set_defaults(lower=False)
parser.add_argument('--m', type=int, default=20, help='MNAV.')
parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--num_epoch', type=int, default=20, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--save_dir', type=str, help='Directory name of the saved model.')
parser.add_argument('--device', type=int, default=0, help='gpu device to use.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--warmup_prop', type=float, default=0.3, help='Proportion of training to perform linear learning rate warmup for.')
parser.add_argument("--eval_per_epoch", default=10, type=int, help="How many times it evaluates on dev set per epoch")
parser.add_argument('--bert', default='bert-base-uncased', type=str, help='Which bert to use.')
parser.add_argument('--train', default='train_episode.json', type=str, help='Which training set to use.')
parser.add_argument('--dev', default='dev_episode.json', type=str, help='Which dev set to use.')
parser.add_argument('--seed', default=42, type=int, help='Random Seed')
parser.add_argument('--hidden_dim', default=2000, type=int)
parser.add_argument('--instances_per_epoch', default=700, type=int)

args = parser.parse_args()
opt = vars(args)

torch.manual_seed(args.seed)
random.seed(args.seed)

tokenizer = BertTokenizer.from_pretrained(opt['bert'])
train_set = EpisodeDataset(opt['data_dir']+opt['train'], tokenizer)
train_batches_size = opt['instances_per_epoch'] // opt['batch_size']
dev_set = EpisodeDataset(opt['data_dir']+opt['train'], tokenizer)
opt['num_training_steps'] = train_batches_size * opt['num_epoch']
opt['num_warmup_steps'] = opt['num_training_steps'] * opt['warmup_prop']
ensure_dir(opt['save_dir'], verbose=True)
eval_step = max(1, train_batches_size // args.eval_per_epoch)
trainer = BERTtrainer(opt)
i = 0
curr_acc = 0
for epoch in range(opt['num_epoch']):
    train_batches = DataLoader(train_set, batch_size=opt['batch_size'], collate_fn=collate_batch, shuffle=True)
    for b in train_batches:
        loss = trainer.update(b)
        if (i + 1) % eval_step == 0:
            # eval on dev
            print("Evaluating on dev set at epoch %d..."%epoch)
            preds = []
            golds = []
            dev_batches = DataLoader(dev_set, batch_size=opt['batch_size'], collate_fn=collate_batch, shuffle=False)
            for db in dev_batches:
                score, loss, labels = trainer.predict(db)
                preds += np.around(score.view(-1).data.cpu().numpy()).tolist()
                golds += labels.view(-1).cpu().tolist()

            matched = [1 if p == golds[j] and p == 1 else 0 for j, p in enumerate(preds)]
            try:
                recall = sum(matched)/sum(golds)
            except ZeroDivisionError:
                recall = 0
            try:
                precision = sum(matched)/sum(preds)
            except ZeroDivisionError:
                precision = 0
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except ZeroDivisionError:
                f1 = 0
            if f1 > curr_acc:
                curr_acc = f1
                trainer.save(opt['save_dir']+'/best_model.pt')
            print ("precision: %f, recall: %f, f1: %f, loss: %f"%(precision, recall, f1, loss))
            print ("current best f1: %f, "%(curr_acc))
        i += 1









    