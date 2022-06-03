import json
from preprocessing_prepeare_sentence import preprocessing
import copy
import sys

model_name = 'bert-base-cased'
pre = preprocessing(model_name)
important_keys = set(['id', 'relation', 'token', 'tokens', 'h', 't', 'head_after_bert', 'tail_after_bert', 'tokens_with_markers', 'head_end', 'tail_end'])

f = json.load(open("dev_episode.json"))

episodes = f[0]

new = [[], f[1]]

for i, ep in enumerate(episodes):
    new[0].append({'meta_train':[], 'meta_test':[]})
    for j, q in enumerate(ep['meta_test']):
        sentence_info = pre.preprocessing_flow(copy.deepcopy(q))
        tokens_with_markers, h_start, t_start, h_end, t_end = sentence_info
        ep['meta_test'][j]["head_after_bert"] = h_start
        ep['meta_test'][j]["tail_after_bert"] = t_start
        ep['meta_test'][j]["tokens_with_markers"] = tokens_with_markers
        ep['meta_test'][j]["head_end"] = h_end
        ep['meta_test'][j]["tail_end"] = t_end

        new_instance = {k:v for k, v in ep['meta_test'][j].items() if k in important_keys}

        new[0][-1]['meta_test'].append(new_instance)

    for j, way in enumerate(ep['meta_train']):
        new[0][-1]['meta_train'].append([])
        for k, shot in enumerate(way):
            sentence_info = pre.preprocessing_flow(copy.deepcopy(shot))
            tokens_with_markers, h_start, t_start, h_end, t_end = sentence_info
            ep['meta_train'][j][k]["head_after_bert"] = h_start
            ep['meta_train'][j][k]["tail_after_bert"] = t_start
            ep['meta_train'][j][k]["tokens_with_markers"] = tokens_with_markers
            ep['meta_train'][j][k]["head_end"] = h_end
            ep['meta_train'][j][k]["tail_end"] = t_end

            new_instance = {k:v for k, v in ep['meta_train'][j][k].items() if k in important_keys}

            new[0][-1]['meta_train'][-1].append(new_instance)

print (json.dumps(new))
            