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
        q["head_after_bert"] = h_start
        q["tail_after_bert"] = t_start
        q["tokens_with_markers"] = tokens_with_markers
        q["head_end"] = h_end
        q["tail_end"] = t_end
    for way in ep['meta_train']:
        for shot in way:
            sentence_info = pre.preprocessing_flow(copy.deepcopy(shot))
            tokens_with_markers, h_start, t_start, h_end, t_end = sentence_info
            q["head_after_bert"] = h_start
            q["tail_after_bert"] = t_start
            q["tokens_with_markers"] = tokens_with_markers
            q["head_end"] = h_end
            q["tail_end"] = t_end

print (json.dumps(f))
            