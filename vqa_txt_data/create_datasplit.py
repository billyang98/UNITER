import json
from lz4.frame import compress, decompress
import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

with open('vqa_words_not_in_bert.txt', 'r') as f:
    data = json.load(f)
with open('vqa_word_questions.json', 'r') as f:
    data2 = json.load(f)

a = set()
for k in data:
    for l in data2[k]:
        a.add(l)
#print(data)

train_set =  lmdb.open('/scratch/cluster/billyang/vqa_dataset/txt_db/oov_datasets/train_set.db', readonly=False, create=True, map_size=100e9)
train_set_c = train_set.begin(write=True)
#train_set_c = txn2.cursor()
valid_set = lmdb.open('/scratch/cluster/billyang/vqa_dataset/txt_db/oov_datasets/valid_set.db', readonly=False, create=True, map_size=10e9)
valid_set_c = valid_set.begin(write=True)
#valid_set_c = txn2.cursor()
test_set =  lmdb.open('/scratch/cluster/billyang/vqa_dataset/txt_db/oov_datasets/test_set.db', readonly=False, create=True, map_size=10e9)
test_set_c = test_set.begin(write=True)
#test_set_c = txn2.cursor()
oov_test_set =  lmdb.open('/scratch/cluster/billyang/vqa_dataset/txt_db/oov_datasets/oov_test_set.db', readonly=False, create=True, map_size=10e9)
oov_test_set_c = oov_test_set.begin(write=True)
#oov_test_set_c = txn2.cursor()

train_images = set()
valid_images = set()
test_images = set()
oov_test_images = set()

train_num_q = 0
valid_num_q = 0
test_num_q = 0
oov_test_num_q = 0

# make deterministic
np.random.seed(0)

# do oov_test_first
for f_name in ['vqa_vg.db', 'vqa_devval.db', 'vqa_train.db', 'vqa_trainval.db']:
    print("Doing {}".format(f_name))
    env = lmdb.open('/scratch/cluster/billyang/vqa_dataset/txt_db/{}'.format(f_name), readonly=True, create=True, map_size=4*1024**4)
    txn = env.begin()
    cursor = txn.cursor()
#    data2 = []
    for key, value in tqdm(cursor):
        q = msgpack.loads(decompress(value))
        img_name = q['img_fname']
        k = key.decode()
        if k in a:
            # only oov questions in this set
            oov_test_set_c.put(key, value)
            oov_test_num_q += 1
            oov_test_images.add(img_name)

oov_test_set_c.commit()

for f_name in ['vqa_vg.db', 'vqa_devval.db', 'vqa_train.db', 'vqa_trainval.db']:
    print("Doing {}".format(f_name))
    env = lmdb.open('/scratch/cluster/billyang/vqa_dataset/txt_db/{}'.format(f_name), readonly=True, create=True, map_size=4*1024**4)
    txn = env.begin()
    cursor = txn.cursor()
#    data2 = []
    for key, value in tqdm(cursor):
        q = msgpack.loads(decompress(value))
        img_name = q['img_fname']
        k = key.decode()
        if k not in a:
            # assign other datasets
            if img_name in oov_test_images or img_name in test_images:
                # image already in a test set
                test_set_c.put(key, value)
                test_num_q +=1
                test_images.add(img_name)
            elif img_name in valid_images:
                valid_set_c.put(key, value)
                valid_num_q +=1
                valid_images.add(img_name)
            elif img_name in train_images:
                train_set_c.put(key, value)
                train_num_q +=1
                train_images.add(img_name)
            else:
                # randomly assign to set
                p = np.random.random()
#            if p < 0.1:
#                test_set_c.put(key, value)
#                test_num_q += 1
#                test_images.add(img_name)
                if p < 0.1:
                    valid_set_c.put(key, value)
                    valid_num_q += 1
                    valid_images.add(img_name)
                else:
                    train_set_c.put(key, value)
                    train_num_q += 1
                    train_images.add(img_name)

test_set_c.commit()
valid_set_c.commit()
train_set_c.commit()

print("Train questions {} images {}".format(train_num_q, len(train_images)))
with open('train_images.json', 'w') as outfile:
    json.dump(list(train_images), outfile)
    
print("Val questions {} images {}".format(valid_num_q, len(valid_images)))
with open('valid_images.json', 'w') as outfile:
    json.dump(list(valid_images), outfile)

print("Test questions {} images {}".format(test_num_q, len(test_images)))
with open('test_images.json', 'w') as outfile:
    json.dump(list(test_images), outfile)

print("OOV Test questions {} images {}".format(oov_test_num_q, len(oov_test_images)))
with open('oov_test_images.json', 'w') as outfile:
    json.dump(list(oov_test_images), outfile)


# Check if sets are disjoint
if not train_images.isdisjoint(valid_images):
    print('train and val images overlap')
if not train_images.isdisjoint(test_images):
    print('train and test images overlap')
if not train_images.isdisjoint(oov_test_images):
    print('train and oov_test images overlap')
if not valid_images.isdisjoint(test_images):
    print('val and test images overlap')
if not valid_images.isdisjoint(oov_test_images):
    print('val and oov test images overlap')

#print(data2)

