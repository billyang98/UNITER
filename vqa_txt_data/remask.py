import json
import lmdb
from pytorch_pretrained_bert import BertTokenizer
import sys
import msgpack
from lz4.frame import compress, decompress



MASK = "[MASK]"
MASK_ID = 103
QUESTION_MARK_ID = 136
QUESTION_MARK= "@@?"

def remask(f_name, out_db, vocab_loc='vqa_words_not_in_bert.txt', strategy='all'):
    with open(vocab_loc, 'r') as f:
        words = json.load(f)
    tok = BertTokenizer.from_pretrained('bert-base-cased')
    def mask_all(query_words):
        query_tokens = []
        for _, word in enumerate(query_words):
            if word in words:
                query_tokens += [MASK]
            else:
                query_tokens += tok.tokenize(word)
        query_ids = tok.convert_tokens_to_ids(query_tokens)
        return query_tokens, query_ids

    if strategy == 'all':
        mask_fn = mask_all
    else:
        print("#### INVALID STRATEGY QUITTING ####")
        return

    new_set =  lmdb.open(out_db, readonly=False, create=True, map_size=10e9)
    new_set_c = new_set.begin(write=True)

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

        query_tokens, query_ids = mask_fn(query_words)
        if query_text[-1] == '?':
            query_tokens.append(QUESTION_MARK)
            query_ids.append(QUESTION_MARK_ID)
        q['toked_question'] = query_tokens
        q['input_ids'] = query_ids
        tokenized_queries[q_id] = q

        new_q = compress(msgpack.dumps(q))
        new_set_c.put(key, new_q)
    new_set_c.commit()



if __name__ == '__main__':
    remask(sys.argv[1], sys.argv[2])
