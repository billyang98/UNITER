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
singularity shell -B <PATH TO UNITER GITHUB>:/uniter,<PATH TO THE VQA DATASET>:/vqa_dataset --nv -w uniter_image
```


# Set up training for VQA
Before running train vqa, we need to move the checkpoint to the correct place to start training.

The training script expects a model at this location and also saves subsequent models to this directory. Thus it is easy to continue training with the same training json file.

Using the same "output_dir" as in your train_<...>.json file:

```
bash scripts/set_up_vqa_training.sh <USERNAME> <OUTPUT_DIR>
```