import json
import lmdb
from pytorch_pretrained_bert import BertTokenizer
import sys
import msgpack
from lz4.frame import compress, decompress
from tqdm import tqdm


MASK = "[MASK]"
MASK_ID = 103
QUESTION_MARK_ID = 136
QUESTION_MARK= "@@?"

def replace_masks_in_q(qid, question_tokens, synonyms_dict, tok):
        new_question_token_ids = []
        synonyms_list = synonyms_dict[qid]
        synonyms_iter = iter(synonyms_list)
        num_masks = 0
        for token in question_tokens:
            if token == MASK_ID:
                num_masks += 1
                new_question_token_ids.append(next(synonyms_iter))
            else:
                new_question_token_ids.append(token)
        assert (num_masks == len(synonyms_list)), 'QID {} has {} masks, but {} found'.format(qid, num_masks, len(synonyms_list))

            
        new_question_tokens = tok.convert_ids_to_tokens(new_question_token_ids)
        return new_question_tokens, new_question_token_ids

def get_synonyms_dict(synonyms_file):
    with open(synonyms_file) as openfile:
        synonyms_list = json.load(openfile)
    synonyms_dict = {}
    for o in synonyms_list:
        qid = o['question_id']
        toks = o['predicted_toks']
        assert (qid not in synonyms_dict), "QID {} already in dict".format(qid)
        synonyms_dict[qid] = toks
    return synonyms_dict

def replace_masks(in_db, out_db, synonyms_file):
    synonyms_dict = get_synonyms_dict(synonyms_file)
    tok = BertTokenizer.from_pretrained('bert-base-cased')

    new_set =  lmdb.open(out_db, readonly=False, create=True, map_size=10e9)
    new_set_c = new_set.begin(write=True)

    env = lmdb.open(in_db, readonly=True, create=False, map_size=4*1024**4)
    txn = env.begin()
    cursor = txn.cursor()

    tokenized_queries = {}

    id2len_dict = {}

    for key, value in tqdm(cursor):
        q_id = key.decode()
        q = msgpack.loads(decompress(value))

        query_tokens, query_ids = replace_masks_in_q(q_id, q['input_ids'], synonyms_dict, tok)
        q['toked_question'] = query_tokens
        q['input_ids'] = query_ids
        tokenized_queries[q_id] = q

        id2len_dict[q_id] = len(query_ids)
        new_q = compress(msgpack.dumps(q))
        new_set_c.put(key, new_q)

    print("committing changes")
    new_set_c.commit()
    with open('{}/id2len.json'.format(out_db), 'w') as outfile:
        json.dump(id2len_dict, outfile)



if __name__ == '__main__':
    replace_masks(sys.argv[1], sys.argv[2], sys.argv[3])
