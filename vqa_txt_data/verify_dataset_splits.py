import json
from lz4.frame import compress, decompress
import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

db_dir = '/scratch/cluster/billyang/vqa_dataset/txt_db/oov_datasets'

with open('test_images.json') as openfile:
    test_images = set(json.load(openfile))
with open('oov_test_images.json') as openfile:
    oov_test_images = set(json.load(openfile))

print(len(oov_test_images.difference(test_images)))
print(len(test_images.difference(oov_test_images)))

for f_name in ['test', 'valid', 'train', 'oov_test']:
    print("Doing {}".format(f_name))
    print('{}/{}_set.db'.format(db_dir, f_name))
    env = lmdb.open('{}/{}_set.db'.format(db_dir, f_name), readonly=True, create=True, map_size=4*1024**4)
    txn = env.begin()
    cursor = txn.cursor()
    img_set = set()
    k_set = set()
    for key, value in tqdm(cursor):
        q = msgpack.loads(decompress(value))
        img_name = q['img_fname']
        k = key.decode()
        k_set.add(k)
        img_set.add(img_name)
    with open('{}_images.json'.format(f_name)) as openfile:
        file_img_set = set(json.load(openfile))
    if img_set != file_img_set:
        print("{} image sets wrong".format(f_name))
    print("{} num questions {} ".format(f_name, len(k_set)))
