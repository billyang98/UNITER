import json

total_json = {}

for img_db_dir in ['coco_train2014', 'coco_val2014', 'vg']:
    print("DOING {}".format(img_db_dir))
    # loop through the original dataset and add to db
    with open('/scratch/cluster/billyang/vqa_dataset/img_db/{}/nbb_th0.2_max100_min10.json'.format(img_db_dir)) as openfile:
        nbb_json = json.load(openfile)

    print("Num images {}".format(len(nbb_json)))

    total_json = {**total_json, **nbb_json}

print("total size {}".format(len(total_json)))

with open('nbb_th0.2_max100_min10.json', 'w') as openfile:
    json.dump(total_json, openfile)

