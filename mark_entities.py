import json
from preprocessing_prepeare_sentence import preprocessing
import copy
import sys

model_name = 'bert-base-cased'
pre = preprocessing(model_name)

f = json.load(open("dev_episode.json"))

episodes = f[0]

for i, ep in enumerate(episodes):
    for j, q in enumerate(ep['meta_test']):
        sentence_info = pre.preprocessing_flow(copy.deepcopy(q))
        tokens_with_markers, h_start, t_start, h_end, t_end = sentence_info
        ep['meta_test'][j]["head_after_bert"] = h_start
        ep['meta_test'][j]["tail_after_bert"] = t_start
        ep['meta_test'][j]["tokens_with_markers"] = tokens_with_markers
        ep['meta_test'][j]["head_end"] = h_end
        ep['meta_test'][j]["tail_end"] = t_end
    for j, way in enumerate(ep['meta_train']):
        for k, shot in enumerate(way):
            sentence_info = pre.preprocessing_flow(copy.deepcopy(shot))
            tokens_with_markers, h_start, t_start, h_end, t_end = sentence_info
            ep['meta_train'][j][k]["head_after_bert"] = h_start
            ep['meta_train'][j][k]["tail_after_bert"] = t_start
            ep['meta_train'][j][k]["tokens_with_markers"] = tokens_with_markers
            ep['meta_train'][j][k]["head_end"] = h_end
            ep['meta_train'][j][k]["tail_end"] = t_end
print (json.dumps(f))
            