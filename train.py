from transformers import BertTokenizer
from dataloader import *
from trainer import BERTtrainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/train_episode.json')
parser.set_defaults(lower=False)
parser.add_argument('--m', type=int, default=1, help='MNAV.')
parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
# parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--device', type=int, default=0, help='gpu device to use.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--warmup_prop', type=float, default=0.3, help='Proportion of training to perform linear learning rate warmup for.')
parser.add_argument("--eval_per_epoch", default=10, type=int, help="How many times it evaluates on dev set per epoch")
parser.add_argument('--bert', default='bert-base-uncased', type=str, help='Which bert to use.')
args = parser.parse_args()
opt = vars(args)

tokenizer = BertTokenizer.from_pretrained(opt['bert'])
ds = EpisodeDataset(opt['data'], tokenizer)
DL_DS = DataLoader(ds, batch_size=opt['batch_size'], collate_fn=collate_batch)
opt['num_training_steps'] = len(DL_DS)
opt['num_warmup_steps'] = opt['num_training_steps'] * opt['warmup_prop']

eval_step = max(1, opt['num_training_steps'] // args.eval_per_epoch)
print (eval_step)
trainer = BERTtrainer(opt)
i = 0
for epoch in range(opt['num_epoch']):
    for b in DL_DS:
        trainer.update(b)

        if (i + 1) % eval_step == 0:
            # eval on dev
            print("Evaluating on dev set...")
            preds = []
            golds = []
            for db in DL_DS:
                score, loss = trainer.predict(b)
                print (score.size())
        i += 1









    