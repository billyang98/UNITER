import json
import csv




qas_file_names = ['qas_synonyms-pg04-mask-2.json', 'qas_mask-pl05-mask-2.json']

for qas_fname in qas_file_names:
    qas_json = json.load(open(qas_fname))
    list_names = ['wtr', 'rtw']
    for l in list_names:
        csv_lists = []
        qs_dict = qas_json[l]
        for qid, q in qs_dict.items():
            csv_row_list = [qid, q['img_fname'], q['bl_q'], q['bl_ans'], q['exp_q'], q['exp_ans']]
            csv_lists.append(csv_row_list)

        with open('{}_{}.json'.format(qas_fname[:-len('.json')], l), 'w') as f:
            write = csv.writer(f)
            write.writerows(csv_lists)