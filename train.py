from transformers import BertTokenizer
from dataloader import *
from trainer import BERTtrainer

opt = {"m":1, 'lr':1e-5, 'warmup_prop':0.2, 'num_warmup_steps': 10, 'num_training_steps':30}
trainer = BERTtrainer(opt)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
ds = EpisodeDataset('../Few_Shot_transformation_and_sampling/train_episode.json', tokenizer)
DL_DS = DataLoader(ds, batch_size=2, collate_fn=collate_batch)
for epoch in range(10):
    for b in DL_DS:
        trainer.update(b)
    