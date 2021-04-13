# Commands for Condor setup


Command to first download the image
Should use scripts/set_up_singularity.sh instead
```
TMPDIR=/scratch/cluster/billyang/bigtmp SINGULARITY_CACHEDIR=/scratch/cluster/billyang/singularity_cache singularity build -s uniter_image docker://chenrocks/uniter
```
```
bash scripts/set_up_singularity.sh <USERNAME>
```

Command to run the image. Input your own paths to the github and where the dataset is saved. We are expecting the dataset directory where the contents are "finetune, img_db, pretrained, txt_db"
```
singularity shell -B /u/billyang/classwork/21s/gnlp/project/UNITER:/uniter,/scratch/cluster/billyang/vqa_dataset:/vqa_dataset --nv -w uniter_image
```


# Set up training for VQA
Before running train vqa, we need to move the checkpoint to the correct place to start training.

The training script expects a model at this location and also saves subsequent models to this directory. Thus it is easy to continue training with the same training json file.

Using the same "output_dir" as in your train_<...>.json file:

```
bash scripts/set_up_vqa_training.sh <USERNAME> <OUTPUT_DIR>
```

How to run the training in a single command

```
/lusr/opt/singularity-3.2.1/bin/singularity exec -B /u/billyang/classwork/21s/gnlp/project/UNITER:/uniter,/scratch/cluster/billyang/vqa_dataset:/vqa_dataset --nv -w /scratch/cluster/billyang/uniter_image bash run_train_singularity.sh train-vqa-mlm-test.json
```

run with condorizer with -n test dry run
```
python scripts/condorizer.py -j TESTING_JOB  -o /scratch/cluster/billyang/ -g  -n singularity exec -B /u/billyang/classwork/21s/gnlp/project/UNITER:/uniter,/scratch/cluster/billyang/vqa_dataset:/vqa_dataset --nv -w /scratch/cluster/billyang/uniter_image bash run_train_singularity.sh train-vqa-mlm-test.jso
```
no -n , not dry run, real submit
```
python scripts/condorizer.py -j TESTING_JOB  -o /scratch/cluster/billyang/condor_output/uniter_vqa -g  singularity exec -B /u/billyang/classwork/21s/gnlp/project/UNITER:/uniter,/scratch/cluster/billyang/vqa_dataset:/vqa_dataset --nv -w /scratch/cluster/billyang/uniter_image bash run_train_singularity.sh train-vqa-mlm-test.jso
```