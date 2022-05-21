from transformers import BertTokenizer
from dataloader import *
from trainer import BERTtrainer
import numpy as np

import argparse
import os

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./')
parser.set_defaults(lower=False)
parser.add_argument('--m', type=int, default=1, help='MNAV.')
parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--num_epoch', type=int, default=10, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--save_dir', type=str, help='Directory name of the saved model.')
parser.add_argument('--device', type=int, default=0, help='gpu device to use.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--warmup_prop', type=float, default=0.3, help='Proportion of training to perform linear learning rate warmup for.')
parser.add_argument("--eval_per_epoch", default=10, type=int, help="How many times it evaluates on dev set per epoch")
parser.add_argument('--bert', default='bert-large-uncased', type=str, help='Which bert to use.')
args = parser.parse_args()
opt = vars(args)

tokenizer = BertTokenizer.from_pretrained(opt['bert'])
train_set = EpisodeDataset(opt['data_dir']+'train_episode_nota_query_only.json', tokenizer)
train_batches = DataLoader(train_set, batch_size=opt['batch_size'], collate_fn=collate_batch)
dev_set = EpisodeDataset(opt['data_dir']+'dev_episode.json', tokenizer)
dev_batches = DataLoader(dev_set, batch_size=1, collate_fn=collate_batch)
opt['num_training_steps'] = len(train_batches) * opt['num_epoch']
opt['num_warmup_steps'] = opt['num_training_steps'] * opt['warmup_prop']
ensure_dir(opt['save_dir'], verbose=True)
eval_step = max(1, opt['num_training_steps'] // args.eval_per_epoch)
trainer = BERTtrainer(opt)
i = 0
curr_acc = 0
for epoch in range(opt['num_epoch']):
    for b in train_batches:
        loss = trainer.update(b)
        if (i + 1) % eval_step == 0:
            # eval on dev
            print("Evaluating on dev set...")
            preds = []
            for db in dev_batches:
                score, loss = trainer.predict(db)
                preds += np.argmax(score.squeeze(2).data.cpu().numpy(), axis=1).tolist()
            nrp = [p == 5 for p in preds]
            nrg = [g == 5 for g in dev_set.get_golds()]
            acc = sum([nrp[i] == nrg[i] for i in range(len(nrp))])/len(nrp)
            if acc > curr_acc:
                curr_acc = acc
                print ("current accuracy: %f"%acc)
                trainer.save(opt['save_dir']+'/best_model.pt')
        i += 1









    