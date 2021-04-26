import json
import lmdb
from pytorch_pretrained_bert import BertTokenizer
import sys
import msgpack
from lz4.frame import compress, decompress
from tqdm import tqdm
from replace_masks import get_synonyms_dict

MASK = "[MASK]"
MASK_ID = 103
QUESTION_MARK_ID = 136
QUESTION_MARK= "@@?"

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def get_qid_synonyms_iter(synonyms_dict, qid):
    if synonyms_dict is not None: 
        if qid in synonyms_dict:
            synonyms_list = synonyms_dict[qid]
            synonyms_iter = iter(synonyms_list)
            return synonyms_iter
        return iter([])
    return None

def replace_token_using_synonyms(word, tok, synonyms_iter, still_mask, mask_low_prob):
    new_tokens = []
    replaced_token = False
    if synonyms_iter is not None:
        try:
            synonym = next(synonyms_iter)
        except StopIteration:
            return [], False
        if synonym == -1:
            # do not use synonym
            if mask_low_prob:
                new_tokens += [MASK]
                replaced_token = True
            else:
                new_tokens += tok.tokenize(word)
        else:
            if mask_low_prob:
                # high probabilty word is tokenized as normal
                new_tokens += tok.tokenize(word)
            elif still_mask:
                # just use mask
                new_tokens += [MASK]
                replaced_token = True
            else:
                new_tokens += tok.convert_ids_to_tokens([synonym])
                replaced_token = True
    else:
        new_tokens += [MASK]
        replaced_token = True
    return new_tokens, replaced_token


def remask(f_name, out_db, vocab_loc='vqa_words_not_in_bert.txt', strategy='all', synonyms_dict=None, mask_low_prob=False, still_mask=False):
    with open(vocab_loc, 'r') as f:
        words = json.load(f)
    tok = BertTokenizer.from_pretrained('bert-base-cased')

    def mask_all(query_words, qid):
        query_tokens = []
        synonyms_iter = get_qid_synonyms_iter(synonyms_dict, qid)
        did_mask = False
        for _, word in enumerate(query_words):
            if word in words:
                tokens_for_word, replaced_token = replace_token_using_synonyms(word, tok, synonyms_iter, still_mask)
                query_tokens += tokens_for_word
                if replaced_token:
                    did_mask = True
            else:
                query_tokens += tok.tokenize(word)
        query_ids = tok.convert_tokens_to_ids(query_tokens)
        return query_tokens, query_ids, did_mask

    def mask_characters(query_words, qid):
        query_tokens = []
        did_mask = False
        synonyms_iter = get_qid_synonyms_iter(synonyms_dict, qid)
        for _, word in enumerate(query_words):
            tokenized_word = tok.tokenize(word)
            if len(tokenized_word) == len(word) and len(word) > 1:
                tokens_for_word, replaced_token = replace_token_using_synonyms(word, tok, synonyms_iter, still_mask, mask_low_prob)
                query_tokens += tokens_for_word
                if replaced_token:
                    did_mask = True
            else:
                query_tokens += tokenized_word
        query_ids = tok.convert_tokens_to_ids(query_tokens)
        return query_tokens, query_ids, did_mask

    def mask_n(n, query_words, qid):
        query_tokens = []
        did_mask = False
        for _, word in enumerate(query_words):
            tokenized_word = tok.tokenize(word)
            if len(tokenized_word) > n:
                query_tokens += [MASK]
                did_mask = True
            else:
                query_tokens += tokenized_word
        query_ids = tok.convert_tokens_to_ids(query_tokens)
        return query_tokens, query_ids, did_mask

    def mask_char3(query_words, qid):
        query_tokens = []
        did_mask = False
        synonyms_iter = get_qid_synonyms_iter(synonyms_dict, qid)
        for _, word in enumerate(query_words):
            tokenized_word = tok.tokenize(word)
            if (len(tokenized_word) == len(word) and len(word) > 1) or len(tokenized_word) > 3:
                tokens_for_word, replaced_token = replace_token_using_synonyms(word, tok, synonyms_iter, still_mask)
                query_tokens += tokens_for_word
                if replaced_token:
                    did_mask = True
            else:
                query_tokens += tokenized_word
        query_ids = tok.convert_tokens_to_ids(query_tokens)
        return query_tokens, query_ids, did_mask
    
    def get_mask_n_lambda(n):
        def mask_n_lambda(query_words, qid):
           return mask_n(n, query_words, qid) 
        return mask_n_lambda

    if strategy == 'all':
        mask_fn = mask_all
    elif strategy == 'character':
        mask_fn = mask_characters
    elif is_int(strategy):
        mask_fn = get_mask_n_lambda(int(strategy)) 
    elif 'char3':
        mask_fn = mask_char3
    else:
        print("#### INVALID STRATEGY QUITTING ####")
        return

    new_set =  lmdb.open(out_db, readonly=False, create=True, map_size=10e9)
    new_set_c = new_set.begin(write=True)

    env = lmdb.open(f_name, readonly=True, create=False, map_size=4*1024**4)
    txn = env.begin()
    cursor = txn.cursor()

    tokenized_queries = {}

    id2len_dict = {}

    questions_changed = []

    for key, value in tqdm(cursor):
        q_id = key.decode()
        q = msgpack.loads(decompress(value))
        original_query_text = q['question']
        query_text = original_query_text
        if query_text[-1] == '?':
            query_text = query_text[0:-1]
        query_words = query_text.split(" ")
        query_words = list(filter(lambda x: x != '', query_words))

        query_tokens, query_ids, did_mask = mask_fn(query_words, q_id)
        if did_mask:
            questions_changed.append(q_id)
            if original_query_text[-1] == '?':
                query_tokens.append(QUESTION_MARK)
                query_ids.append(QUESTION_MARK_ID)
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
    with open('{}/questions_changed.json'.format(out_db), 'w') as outfile:
        json.dump(questions_changed, outfile)



if __name__ == '__main__':
    synonyms_dict = None
    mask_low_prob = False
    still_mask = False
    if len(sys.argv) > 4:
        synonyms_dict = get_synonyms_dict(sys.argv[4])
    if len(sys.argv) > 5:
        mask_low_prob = True
    if len(sys.argv) > 6:
        still_mask = True

    remask(sys.argv[1], sys.argv[2], strategy=sys.argv[3], synonyms_dict=synonyms_dict, mask_low_prob=mask_low_prob, still_mask=still_mask)
