import json
from lz4.frame import compress, decompress
import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

f_name = '/scratch/cluster/billyang/vqa_dataset/txt_db/oov_datasets/oov_test_set.db'
print("Doing {}".format(f_name))
env = lmdb.open(f_name, readonly=True, create=True, map_size=4*1024**4)
txn = env.begin()
cursor = txn.cursor()

ans2label = json.load(open('../utils/ans2label.json'))
label2ans = {label: ans for ans, label in ans2label.items()}

id_ans = {}
for key, value in tqdm(cursor):
    q = msgpack.loads(decompress(value))
    k = key.decode()

    answer_ids = q['target']['labels']
    scores = q['target']['scores']
    id_ans[k] = {'ids': answer_ids, 'strings': [label2ans[a] for a in answer_ids], 'scores': scores}

json.dump(id_ans, open('oov_test_full_answers.json', 'w'))

