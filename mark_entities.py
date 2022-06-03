import json
from preprocessing_prepeare_sentence import preprocessing

model_name = 'bert-base-cased'
pre = preprocessing(bert_model)
for i, ep in enumerate(episodes):
    for j, q in enumerate(ep['meta_test']):
        sentence_info = pre.preprocessing_flow(copy.deepcopy(q))
        print (sentence_info)
    for way in ep['meta_train']:
        for shot in way:
            sentence_info = pre.preprocessing_flow(copy.deepcopy(shot))
            print (sentence_info)
            