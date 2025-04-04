USERNAME=$1
OUTPUT_DIR=$2
DATASET_DIR=$3

cp scripts/run_train_singularity.sh /scratch/cluster/$USERNAME/uniter_image/
cp scripts/run_eval_singularity.sh /scratch/cluster/$USERNAME/uniter_image/

cd /scratch/cluster/$USERNAME/uniter_image

mkdir -p uniter
mkdir -p vqa_dataset
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/ckpt
if [ ! -f  $OUTPUT_DIR/ckpt/model_step_0.pt ] ; then
  cp $DATASET_DIR/pretrained/uniter-base.pt $OUTPUT_DIR/ckpt/model_step_0.pt
fi
