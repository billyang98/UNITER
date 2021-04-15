import json
from lz4.frame import compress, decompress
import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

dbs_dir = '/scratch/cluster/billyang/vqa_dataset/txt_db/oov_datasets'
for f_name in ['test', 'valid', 'train', 'oov_test']:
    print("Doing {}".format(f_name))
    db_dir = '{}/{}_set.db'.format(dbs_dir, f_name)
    print(db_dir)
    env = lmdb.open(db_dir, readonly=True, create=True, map_size=4*1024**4)
    txn = env.begin()
    cursor = txn.cursor()

    id2len_dict = {}
    txt2img_dict = {}
    img2txt_dict = {}
    for key, value in tqdm(cursor):
        q = msgpack.loads(decompress(value))
        k = key.decode()

        id_len = len(q['input_ids'])
        id2len_dict[k] = id_len

        img_name = q['img_fname'] 
        txt2img_dict[k] = img_name

        if img_name not in img2txt_dict:
            img2txt_dict[img_name] = [k]
        else:
            img2txt_dict[img_name].append(k)

    # Verify stuff
    print('\n\n==== {} INFO ===='.format(f_name))
    print('id2len dict length {}'.format(len(id2len_dict)))
    print('txt2img dict length {}'.format(len(txt2img_dict)))
    print('img2txt dict length {}\n\n'.format(len(img2txt_dict)))

    for txt in txt2img_dict:
        img = txt2img_dict[txt]
        if img not in img2txt_dict:
            print("img {} not in img2txt".format(img))
        elif txt not in img2txt_dict[img]:
            print("txt {} not in img2txt[img {}]".format(txt, img))

     # Write out
    with open('{}/id2len.json'.format(db_dir), 'w') as outfile:
        json.dump(id2len_dict, outfile)

    with open('{}/txt2img.json'.format(db_dir), 'w') as outfile:
        json.dump(txt2img_dict, outfile)

    with open('{}/img2txts.json'.format(db_dir), 'w') as outfile:
        json.dump(img2txt_dict, outfile)




