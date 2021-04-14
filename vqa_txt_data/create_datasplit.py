import json
from lz4.frame import compress, decompress
import lmdb
import msgpack
import numpy as np

with open('vqa_words_not_in_bert.txt', 'r') as f:
    data = json.load(f)
with open('vqa_word_questions.json', 'r') as f:
    data2 = json.load(f)

a = set()
for k in data:
    for l in data2[k]:
        a.add(l)
print(data)

train_set =  lmdb.open('train_set.db', readonly=False, create=True, map_size=100e9)
txn2 = train_set.begin(write=True)
train_set_c = txn2.cursor()
valid_set = lmdb.open('valid_set.db', readonly=False, create=True, map_size=10e9)
txn2 = valid_set.begin(write=True)
valid_set_c = txn2.cursor()
test_set =  lmdb.open('test_set.db', readonly=False, create=True, map_size=10e9)
txn2 = test_set.begin(write=True)
test_set_c = txn2.cursor()
oov_test_set =  lmdb.open('test_set.db', readonly=False, create=True, map_size=10e9)
txn2 = oov_test_set.begin(write=True)
oov_test_set_c = txn2.cursor()



for f_name in ['vqa_vg.db', 'vqa_devval.db', 'vqa_train.db', 'vqa_trainval.db']:
    env = lmdb.open('txt_db/{}'.format(f_name), readonly=False, create=True, map_size=4*1024**4)
    txn = env.begin()
    cursor = txn.cursor()
    data2 = []
    for key, value in cursor:
        q = msgpack.loads(decompress(value))
        k = key.decode()
        if k in a:
            oov_test_set_c.put(key, value)
        else:
            p = np.random.random()
            if p < 0.1:
                test_set_c.put(key, value)
            elif p < 0.2:
                valid_set_c.put(key, value)
            else:
                train_set_c.put(key, value)
    

#print(data2)

