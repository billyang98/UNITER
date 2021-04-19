import json
import lmdb
from pytorch_pretrained_bert import BertTokenizer
import sys
import msgpack
from lz4.frame import compress, decompress



MASK = "[MASK]"



def remask(f_name, vocab_loc='vqa_words_not_in_bert.txt'):
    with open(vocab_loc, 'r') as f:
        words = json.load(f)
    
    tok = BertTokenizer.from_pretrained('bert-base-cased')
    
    
    env = lmdb.open(f_name, readonly=True, create=False, map_size=4*1024**4)
    txn = env.begin()
    cursor = txn.cursor()
    
    tokenized_queries = {}
    
    for key, value in cursor:
        q_id = key.decode()
        q = msgpack.loads(decompress(value))
        query_text = q['question']
        if query_text[-1] == '?':
            query_text = query_text[0:-1]
        query_words = query_text.split(" ")
        query_tokens = []
        for i, word in enumerate(query_words):
            if word in words:
                query_tokens += [MASK]
            else:
                query_tokens += tok.tokenize(word)
        q['new_toked_question'] = query_tokens
        tokenized_queries[q_id] = q
    return tokenized_queries

if __name__ == '__main__':
    tokenized_queries = remask(sys.argv[1])
    print(tokenized_queries)
