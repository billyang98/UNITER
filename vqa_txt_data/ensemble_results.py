import json
import numpy as np
from tqdm import tqdm



ensemble_answers = {}
for ensemble in tqdm(range(1, 6)):
    exp_name = 'results_test_synonyms_mask_2_ensemble_{}'.format(ensemble)
    exp_ans_file = '/scratch/cluster/billyang/uniter_image/vqa_joint_trained/{}/results_3000_all.json'.format(exp_name)
    exp_ans_list = json.load(open(exp_ans_file))
    exp_ans = {o['question_id']: o['answer'] for o in exp_ans_list}

    for qid, answer in exp_ans.items():
        if qid not in ensemble_answers:
            ensemble_answers[qid] = [answer]
        else:
            ensemble_answers[qid].append(answer)

ensemble_answer = []
for qid, answers in tqdm(ensemble_answers.items()):
    answers_dict = {}
    for answer in answers:
        if answer not in answers_dict:
            answers_dict[answer] = 1
        else:
            answers_dict[answer] += 1
    answers_list = [(k, v) for k,v in answers_dict.items()]
    answers_list.sort(key=lambda x:x[1], reverse=True)
    top_answer_set = set()
    top_count = 0
    for answer, count in answers_list:
        if count >= top_count:
            top_answer_set.add(answer)
            top_count = count
    for answer in answers:
        if answer in top_answer_set:
            ensemble_answer.append({'question_id': qid, 'answer': answer})
            break

json.dump(ensemble_answer, open('results_test_synonyms_mask_2_ensemble_all_5.json', 'w'))
