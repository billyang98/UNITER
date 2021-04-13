USERNAME=$1
OUTPUT_DIR=$2

cd /scratch/cluster/$USERNAME/uniter_image
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/ckpt
cp vqa_dataset/pretrained/uniter-base.pt $OUTPUT_DIR/ckpt/model_step_0.pt