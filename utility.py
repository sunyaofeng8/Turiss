import tensorflow as tf
import tensorflow_hub as hub

import bert
import numpy as np

class BertTokenizer:
    def __init__(self, max_len, bert_layer):
        self.FullTokenizer = bert.bert_tokenization.FullTokenizer
        self.bert_layer = bert_layer
        self.vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = self.FullTokenizer(self.vocab_file, self.do_lower_case)
        self.max_len = max_len

    def get_masks(self, tokens, max_len):
        """Mask for padding"""    
        if len(tokens)>max_len:    
            print(len(tokens))    
            raise IndexError("Token length more than max seq length!")    
        return [1]*len(tokens) + [0] * (max_len - len(tokens)) 

    def get_segments(self, tokens, max_len):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens)>max_len:
            print(len(tokens))
            raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_len - len(tokens))

    def get_ids(self, tokens, tokenizer, max_len):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (max_len-len(token_ids))
        return input_ids

    def GetStokens(self, s):
        stokens = self.tokenizer.tokenize(s)
        stokens = stokens[:self.max_len-2]
        stokens = ["[CLS]"] + stokens + ["[SEP]"]

        return stokens
    
    def GetInput_ids(self, stokens):
        return self.get_ids(stokens, self.tokenizer, self.max_len)

    def GetInput_masks(self, stokens):
        return self.get_masks(stokens, self.max_len)
    
    def GetInput_segments(self, stokens):
        return self.get_segments(stokens, self.max_len)

class CleanedTextDict:
    def __init__(self, trainset, testset):
        count_of_words = {}
        words = (' '.join(list(trainset['CleanedText'].values) + list(testset['CleanedText'].values))).split()
        
        for word in words:
            if not word in count_of_words:
                count_of_words[word] = 1
            else:
                count_of_words[word] += 1
        
        self.dic = {}
        self.VocabSize = 0
        self.threshold = 10

        for word in count_of_words:
            if count_of_words[word] <= 10: # Nan Threshold
                continue
            self.dic[word] = self.VocabSize
            self.VocabSize += 1
        
        print("Cleaned Text Dict, Threshold = %d, Vocab Size = %d" % (self.threshold, self.VocabSize))
    
    def TextToStr(self, raw_X):
        X = raw_X.split()
        X = [t for t in X if t in self.dic]
        X = X[:120]
        X = ' '.join(X)
        return X


        '''
        X = [x.split() for x in raw_X]
        X = [[self.dic[t] for t in x if t in self.dic] for x in X]

        max_len = max([len(x) for x in X])
        X = [' '.join(x) for x in X]

        return max_len, X
        '''


if __name__ == "__main__":
    s = "Hello World"
    max_len = 10

    tokenizer = BertTokenizer('./bert_layer')
    input_ids, input_masks, input_segments = tokenizer.StrToTokens(s, max_len)

    print(input_ids)
    print(input_masks)
    print(input_segments)



'''
from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow

input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("./bert_layer",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

pool_embs, all_embs = model.predict([np.array([input_ids]),np.array([input_masks]),np.array([input_segments])])

print(pool_embs)
'''