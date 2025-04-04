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
python scripts/condorizer.py -j UNITER_VQA_OOV_1  -o /scratch/cluster/billyang/condor_output -g  -n /lusr/opt/singularity-3.2.1/bin/singularity exec -B /u/billyang/classwork/21s/gnlp/project/UNITER:/uniter,/scratch/cluster/billyang/vqa_dataset:/vqa_dataset --nv -w /scratch/cluster/billyang/uniter_image bash run_train_singularity.sh train_vqa_joint.json
```
no -n , not dry run, real submit
```
python scripts/condorizer.py -j UNITER_VQA_OOV_1  -o /scratch/cluster/billyang/condor_output -g /lusr/opt/singularity-3.2.1/bin/singularity exec -B /u/billyang/classwork/21s/gnlp/project/UNITER:/uniter,/scratch/cluster/billyang/vqa_dataset:/vqa_dataset --nv -w /scratch/cluster/billyang/uniter_image bash run_train_singularity.sh train-vqa-joint.json
```

condor with eval
```
python scripts/condorizer.py -j UNITER_VQA_EVAL_OOV_NORMAL  -o /scratch/cluster/billyang/condor_output -g  -n /lusr/opt/singularity-3.2.1/bin/singularity exec -B /u/billyang/classwork/21s/gnlp/project/UNITER:/uniter,/scratch/cluster/billyang/vqa_dataset:/vqa_dataset --nv -w /scratch/cluster/billyang/uniter_image bash run_eval_singularity.sh eval-vqa.json
```

condor with inf
```
python scripts/condorizer.py -j UNITER_VQA_EVAL_MASKED_CHAR  -o /scratch/cluster/billyang/condor_output -g  /lusr/opt/singularity-3.2.1/bin/singularity exec -B /u/billyang/classwork/21s/gnlp/project/UNITER:/uniter,/scratch/cluster/billyang/vqa_dataset:/vqa_dataset --nv -w /scratch/cluster/billyang/uniter_image python uniter/inf_mlm_vqa.py --config uniter/config/inf-vqa-masked-char.json
```


install horovod
```
 HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1  pip install --no-cache-dir horovod


HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 \
>     pip install --no-cache-dir horovod


```

install pytorch
```
conda install pytorch=1.7.0 torchvision cudatoolkit=10.1 -c pytorch
```