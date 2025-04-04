import json
import numpy as np
from tqdm import tqdm

# Change these based on experiment
#exp_dataset = 'mask_char_oov_test_set.db'
#exp_name = 'results_test_mask_char'
#exp_dataset = 'mask_2_oov_test_set.db'
#exp_name = 'results_test_mask_2'
#exp_dataset = 'mask_2_oov_test_set.db'
#exp_name = 'results_test_synonyms_mask_2_ensemble_all_5'

#exp_dataset = 'synonyms_mask_char_l03_oov_test_set.db'
#exp_name = 'results_test_synonyms_mask_char_l03'
#exp_dataset = 'synonyms_mask_char_03m_oov_test_set.db'
#exp_name = 'results_test_synonyms_mask_char_03m'
#exp_dataset = 'synonyms_mask_2_03l_oov_test_set.db'
#exp_name = 'results_test_synonyms_mask_2_03l'

exp_dataset = 'mask_2_oov_test_set.db'
exp_name = 'results_test_synonyms_mask_2_fixed'


q_list_file = '/scratch/cluster/billyang/vqa_dataset/txt_db/oov_datasets/{}/questions_changed.json'.format(exp_dataset)
exp_ans_file = '/scratch/cluster/billyang/uniter_image/vqa_joint_trained/{}/results_3000_all.json'.format(exp_name)
#exp_ans_file = '/scratch/cluster/billyang/uniter_image/vqa_joint_fixed_trained/{}/results_3000_all.json'.format(exp_name)

q_list = json.load(open(q_list_file))

exp_ans_list = json.load(open(exp_ans_file))
baseline_ans_list = json.load(open('/scratch/cluster/billyang/uniter_image/vqa_joint_trained/results_test_normal_test/results_3000_all.json'))
#baseline_ans_list = json.load(open('/scratch/cluster/billyang/uniter_image/vqa_joint_fixed_trained/results_test_normal_test_fixed/results_3000_all.json'))
exp_ans = {o['question_id']: o['answer'] for o in exp_ans_list}
baseline_ans = {o['question_id']: o['answer'] for o in baseline_ans_list}

gt_ans = json.load(open('oov_test_full_answers.json'))

results = {}
results['num_questions'] = len(q_list)
exp_tot_score = 0
bl_tot_score = 0
rtw = []
wtr = []

def getscore(answer, answers, scores):
    if answer in answers:
        return scores[answers.index(answer)]
    return 0

for qid in tqdm(q_list):
    exp_score = getscore(exp_ans[qid], gt_ans[qid]['strings'], gt_ans[qid]['scores'])
    exp_tot_score += exp_score
    bl_score = getscore(baseline_ans[qid], gt_ans[qid]['strings'], gt_ans[qid]['scores'])
    bl_tot_score += bl_score

    if exp_score > 0 and bl_score == 0:
        wtr.append(qid)
    if bl_score > 0 and exp_score == 0:
        rtw.append(qid)

results['exp_score'] = exp_tot_score / len(q_list)
results['bl_score'] = bl_tot_score / len(q_list)
results['rtw'] = rtw
results['wtr'] = wtr
results['rtw_count'] = len(rtw)
results['wtr_count'] = len(wtr)

print("dumping")
json.dump(results, open('{}.json'.format(exp_name), 'w'))




    

# get new scores



# find answers wrong to right

# find answers right to wrong
