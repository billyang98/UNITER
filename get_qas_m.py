import json
import lmdb
from pytorch_pretrained_bert import BertTokenizer
import sys
import msgpack
from lz4.frame import compress, decompress
from tqdm import tqdm
import random 

class TxtLmdb(object):
    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir,
                                 readonly=True, create=False)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                             raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret

random.seed(0)

bl_db_path = '/scratch/cluster/billyang/vqa_dataset/txt_db/oov_datasets/oov_test_set.db'
baseline_ans_list = json.load(open('/scratch/cluster/billyang/uniter_image/vqa_joint_trained/results_test_normal_test/results_3000_all.json'))
bl_ans = {o['question_id']: o['answer'] for o in baseline_ans_list}

mexp_db_path = '/scratch/cluster/billyang/vqa_dataset/txt_db/oov_datasets/synonyms_mask_2_04m_oov_test_set.db'
mexp_ans_list = json.load(open('/scratch/cluster/billyang/uniter_image/vqa_joint_trained/results_test_synonyms_mask_2_04m/results_3000_all.json'))
mexp_ans = {o['question_id']: o['answer'] for o in baseline_ans_list}


def get_qas(name, exp_db_path, exp_ans_path, exp_comp_path, sample=-1):
    exp_ans_list = json.load(open(exp_ans_path))
    exp_ans = {o['question_id']: o['answer'] for o in exp_ans_list}

    exp_comp = json.load(open(exp_comp_path))
    rtw_list = exp_comp['rtw']
    wtr_list = exp_comp['wtr']
    if sample >= 0:
        rtw_list = random.sample(rtw_list, min(sample, len(rtw_list)))
        wtr_list = random.sample(wtr_list, min(sample, len(wtr_list)))

    bl_db = TxtLmdb(bl_db_path)
    exp_db = TxtLmdb(exp_db_path)
    mexp_db = TxtLmdb(mexp_db_path)

    qas_dict = {'rtw': {}, 'wtr': {}}

    for list_name, q_list in [('rtw', rtw_list), ('wtr', wtr_list)]:
        for qid in q_list:
            bl_value = bl_db[qid]
            exp_value = exp_db[qid]
            mexp_value = mexp_db[qid]

            bl_a = bl_ans[qid]
            exp_a = exp_ans[qid]
            bl_q = bl_value['toked_question']
            exp_q = exp_value['toked_question']
            img_fname = bl_value['img_fname']

            qas_dict[list_name][qid] = {'bl_q': bl_q, 'exp_q': exp_q, 'mexp_q':
            mexp_value['toked_question'], 'bl_ans': bl_a, 'exp_ans': exp_a,
            'mexp_ans': mexp_ans[qid], 'img_fname': img_fname}
    print("dumping")
    json.dump(qas_dict, open('qas_{}.json'.format(name), 'w'))
        

if __name__ == '__main__':
    if len(sys.argv) == 6:
        sample = int(sys.argv[5])
    else:
        sample = -1
    get_qas(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sample)
