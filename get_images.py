import json
import lmdb
import msgpack
from tqdm import tqdm
import shutil

#env = lmdb.open('/scratch/cluster/billyang/vqa_dataset/img_db/oov_datasets/oov_test_set/feat_th0.2_max100_min10', readonly=True, create=True, map_size=4*1024**4, writemap=True)
#txn = env.begin()
#name2nbb = json.load(open('/scratch/cluster/billyang/vqa_dataset/img_db/oov_datasets/oov_test_set/nbb_th0.2_max100_min10.json'))
#
##files = ['questions_answers/qas_mask-pl05-mask-2.json', 'questions_answers/qas_synonyms-pg04-mask-2.json']
#files = ['questions_answers/qas_ms_2_pg04.json']
#   
#img_fnames = set()
#
#for file_name in files:
#    qas = json.load(open(file_name))
#    for q_dict in qas.values():
#        for q in q_dict.values():
#            img_fnames.add(q['img_fname'])
#
#json.dump(list(img_fnames), open('img_list2.json','w'))

#for img_fname in img_fnames:
#    nbb = name2nbb[img_fname]
#    dump = txn.get(img_fname.encode('utf-8'))
#    img_dump = msgpack.loads(dump, raw=False)
##    img_feat = img_dump['norm_bb']['data'.encode('utf-8')]
#    img_feat = img_dump['features'].keys()
#    print(img_feat)
#    break
        
img_fnames = json.load(open('img_list2.json'))
for img_fname in img_fnames:
    if img_fname.startswith('coco_val2014'):
        img_num = img_fname[len('coco_val2014_'): -len('.npz')]
        img_path = '/scratch/cluster/billyang/vqa_images_raw/val2014/COCO_val2014_{}.jpg'.format(img_num)
    elif img_fname.startswith('coco_train2014'):
        img_num = img_fname[len('coco_train2014_'): -len('.npz')]
        img_path = '/scratch/cluster/billyang/vqa_images_raw/train2014/COCO_train2014_{}.jpg'.format(img_num)
    elif img_fname.startswith('vg'):
        img_num = int(img_fname[len('vg_'): -len('.npz')])
        img_path = '/scratch/cluster/billyang/vqa_images_raw/vg/{}.jpg'.format(img_num)
    else:
        print('invalid img_fname {}'.format(img_fname))
    print(img_fname)
    print(img_path)
    shutil.copyfile(img_path, 'vqa_images2/{}'.format(img_fname))


