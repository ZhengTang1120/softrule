import torch
from torch.utils.data import Dataset, DataLoader
import json
from collections import defaultdict


ENTITY_TOKEN_TO_ID = {'[OBJ-CAUSE_OF_DEATH]': 3, '[OBJ-CITY]': 2, '[OBJ-DATE]': 17, '[OBJ-PERSON]': 14, '[OBJ-URL]': 9, '[OBJ-NATIONALITY]': 16, '[OBJ-ORGANIZATION]': 18, '[OBJ-MISC]': 11, '[OBJ-NUMBER]': 12, '[OBJ-CRIMINAL_CHARGE]': 7, '[SUBJ-ORGANIZATION]': 0, '[SUBJ-PERSON]': 1, '[OBJ-DURATION]': 4, '[OBJ-COUNTRY]': 8, '[OBJ-LOCATION]': 15, '[OBJ-RELIGION]': 10, '[OBJ-TITLE]': 6, '[OBJ-STATE_OR_PROVINCE]': 5, '[OBJ-IDEOLOGY]': 13}
PAD_ID = 0

class EpisodeDataset(Dataset):
    def __init__(self, filename, tokenizer, train_rels):
        super(EpisodeDataset).__init__()
        f = json.load(open(filename))
        self.tokenizer = tokenizer
        self.query_size = len(f[2][0][1])
        self.parse(f[0], f[2])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        query = self.queries[idx]
        support_sents = self.support_sents[idx]
        label = self.labels[idx]
        return {'query':query, 'support_sents':support_sents, 'label':label} 

    def get_golds(self):
        return self.labels    

    def parse(self, episodes, labels):
        self.queries = list()
        self.support_sents = list()
        self.labels = list()
        for i, ep in enumerate(episodes):
            for j, q in enumerate(ep['meta_test']):
                self.queries.append(self.parseTACRED(q))
                self.labels.append(1 if labels[i][1][j] in train_rels else 0)
                self.support_sents.append([])
                for way in ep['meta_train']:
                    self.support_sents[-1].append([])
                    for shot in way:
                        self.support_sents[-1][-1].append(self.parseTACRED(shot))
            

    def parseTACRED(self, instance):
        words = list()
        ss, se = instance['subj_start'], instance['subj_end']
        os, oe = instance['obj_start'], instance['obj_end']

        for i, t in enumerate(instance['token']):
            if i == ss:
                words.append("[unused%d]"%(ENTITY_TOKEN_TO_ID['[SUBJ-'+instance['subj_type']+']']))
            if i == os:
                words.append("[unused%d]"%(ENTITY_TOKEN_TO_ID['[OBJ-'+instance['obj_type']+']']))
            if i>=ss and i<=se:
                pass
            elif i>=os and i<=oe:
                pass
            else:
                t = convert_token(t)
                for j, sub_token in enumerate(self.tokenizer.tokenize(t)):
                    words.append(sub_token)
        
        words = ['[CLS]'] + words + ['[SEP]']
        tokens = self.tokenizer.convert_tokens_to_ids(words)
        return tokens

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
            return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token

def pad_list(tokens_list, token_len=None):
    if token_len is None:
        token_len = max([len(x) for x in tokens_list])
    pad_tokens_list = [[0 for _ in range(token_len)] for _ in tokens_list]
    for i, t in enumerate(tokens_list):
        pad_tokens_list[i][:len(t)] = t
    return pad_tokens_list

def collate_batch(batch):
    queries = list()
    support_sents = list()
    labels = list()
    max_ss_l = max([max([max([len(s) for s in ss]) for ss in d['support_sents']]) for d in batch])
    for d in batch:
        queries.append(d['query'])
        support_sents.append([])
        for ss in d['support_sents']:
            support_sents[-1].append(pad_list(ss, max_ss_l))
        labels.append(d['label'])
    return torch.LongTensor(pad_list(queries)), torch.LongTensor(support_sents), torch.LongTensor(labels)


