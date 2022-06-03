from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
# from my_library.dataset_readers.mtb_reader import head_start_token,head_end_token,tail_start_token,tail_end_token
# from my_library.models.my_bert_tokenizer import MyBertWordSplitter
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter, SimpleWordSplitter

entity_replace_token = '*'

head_start_token = '[unused1]'  # fixme check if this indeed the token they used in the paper
head_end_token = '[unused2]'  # fixme check if this indeed the token they used in the paper
tail_start_token = '[unused3]'
tail_end_token = '[unused4]'


def encode_by_bert(sent ,tokenizer):
    my_sen = tokenizer.convert_tokens_to_string(sent)
    input_ids = tokenizer.encode(text=my_sen, add_special_tokens=True)
    return input_ids


class preprocessing(object):
    def __init__(self,bert_model):
        lower_case = True if "uncased" in bert_model else False
        self.bert_indexer,self.tokenizer = self.get_bert_indexer(bert_model,lower_case=lower_case)
        # self.tokenizer_bert = MyBertWordSplitter(do_lower_case=lower_case)
        self.spacy_splitter = SpacyWordSplitter(keep_spacy_tokens=True)
        self.just_space_tokenization = JustSpacesWordSplitter()
        self.simple_tokenization = SimpleWordSplitter()

    def preprocessing_flow(self,relation):
        sentence, head, tail = relation["tokens"], relation["h"], relation["t"]
        if len(set([head_start_token,head_end_token,tail_start_token,tail_end_token]).intersection(set(sentence))) != 4:
            self.addStartEntityTokens(sentence, head, tail)

        bert_indexing = self.bert_indexer(sentence)
        head_location_after_bert,tail_location_after_bert,head_end,tail_end = self.get_h_t_index_after_bert(bert_indexing)
        return [sentence,head_location_after_bert,tail_location_after_bert,head_end,tail_end]

    def reduce_to_single_entity_span(self,head,tail):
        if len(head[2]) > 1 and len(tail[2]) > 1:
            minimal = 999
            for head_span in head[2]:
                tail_span,this_head_span_minimal_distance = self.find_nearest_entity_to_span(tail, head_span)
                if this_head_span_minimal_distance < minimal:
                    minimal = this_head_span_minimal_distance
                    head_span_to_be_chosen = head_span
                    tail_span_to_be_chosen = tail_span
            head[2] = [head_span_to_be_chosen]
            tail[2] = [tail_span_to_be_chosen]



        elif len(head[2]) > 1:
            head_span,_ = self.find_nearest_entity_to_span(head, tail[2][0])
            head[2] = [head_span]
        elif len(tail[2]) > 1:
            tail_span,_ = self.find_nearest_entity_to_span(tail, head[2][0])
            tail[2] = [tail_span]


    def find_nearest_entity_to_span(self, e1, e2_span):
        e2_start = e2_span[0]
        e2_end = e2_span[-1]
        minimal = 999
        for h in e1[2]:
            e1_start, e1_end = h[0], h[-1]
            this_minimal_dis = min(abs(e1_start - e2_end), abs(e1_end - e2_start))
            if this_minimal_dis < minimal:
                e1_span_to_be_return = h
                minimal = this_minimal_dis
        return e1_span_to_be_return,minimal

    def addStartEntityTokens(self, tokens_list, head_full_data, tail_full_data):
        self.reduce_to_single_entity_span(head_full_data,tail_full_data)
        head_start_location, head_end_location = self.return_locations(head_full_data)
        tail_start_location, tail_end_location = self.return_locations(tail_full_data)

        h_start_location, head_end_location, tail_start_location, tail_end_location = head_start_location[0],head_end_location[0], tail_start_location[0], tail_end_location[0]

        offset_tail = 2 * (tail_start_location > h_start_location)
        tokens_list.insert(h_start_location, head_start_token)  # arbetrary pick a token for that
        tokens_list.insert(head_end_location + 1 + 1, head_end_token)  # arbetrary pick a token for that
        tokens_list.insert(tail_start_location + offset_tail, tail_start_token)  # arbetrary pick a token for that
        tokens_list.insert(tail_end_location + 2 + offset_tail, tail_end_token)  # arbetrary pick a token for that

        return h_start_location + 2 - offset_tail, tail_start_location + offset_tail

    def return_locations(self, head_full_data):
        end_location, start_location = [head_full_data[2][0][-1]], [head_full_data[2][0][0]]
        return start_location, end_location


    def get_bert_indexer(self, bert_model_this_model,lower_case):
        from pytorch_transformers import BertModel, BertTokenizer
        my_special = {'additional_special_tokens': [head_start_token, head_end_token, tail_start_token, tail_end_token]}
        model_class, tokenizer_class, pretrained_weights = BertModel, BertTokenizer, bert_model_this_model
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights,do_lower_case=lower_case,
                                                    never_split=[head_start_token, head_end_token, tail_start_token, tail_end_token])
        tokenizer.add_special_tokens(my_special)
        return lambda x: encode_by_bert(x,tokenizer),tokenizer

    def get_h_t_index_after_bert(self, bert_indexing):
        return bert_indexing.index(1), bert_indexing.index(3) , bert_indexing.index(2), bert_indexing.index(4)




