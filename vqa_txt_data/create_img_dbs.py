import json
from lz4.frame import compress, decompress
import lmdb
import msgpack
import numpy as np
from tqdm import tqdm
from os.path import exists
from os import makedirs

dbs_dir = '/scratch/cluster/billyang/vqa_dataset/img_db/oov_datasets'

for split in ['test', 'valid', 'train', 'oov_test']:
    print("\n DOING {}".format(split))
    db_dir = '{}/{}_set'.format(dbs_dir, split)
    print(db_dir)
    if not exists(db_dir):
        makedirs(db_dir)

    with open('{}_images.json'.format(split)) as openfile:
        img_set = json.load(openfile)

    print("Image set length {}".format(len(img_set)))

    # get write cursor for a single split
    lmdb_set =  lmdb.open('{}/feat_th0.2_max100_min10'.format(db_dir), readonly=False, create=True, map_size=100e9)
    lmdb_set_c = lmdb_set.begin(write=True)

    added_imgs = [] 

    for img_lmdb in ['coco_train2014', 'coco_val2014', 'vg']:
        print("    DOING {}".format(img_lmdb))
        # loop through the original dataset and add to db
        env = lmdb.open('/scratch/cluster/billyang/vqa_dataset/img_db/{}/feat_th0.2_max100_min10'.format(img_lmdb), readonly=True, create=True, map_size=4*1024**4, writemap=True)
        txn = env.begin()
        
        
        cursor = txn.cursor()
        print('    Total entries {}'.format(env.stat()['entries']))
        for key, value in tqdm(cursor):
            k = key.decode()
            if k in img_set:
                lmdb_set_c.put(key, value)
                added_imgs.append(k)
        
        # loop through the set and add if we find if in this db
#        for img_name in tqdm(img_set):
#            value = txn.get(img_name.encode('utf-8'))
#            if value is not None:
#                lmdb_set_c.put(img_name.encode('utf-8'), value)
#                added_imgs.append(img_name)


    lmdb_set_c.commit()

    print("Total imgs added {}".format(len(added_imgs)))
    print("Total unique imgs added {}".format(len(set(added_imgs))))



