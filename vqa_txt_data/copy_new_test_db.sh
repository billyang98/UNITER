USERNAME=$1
OUTDIR=$2

cp /scratch/cluster/$USERNAME/vqa_dataset/txt_db/oov_datasets/oov_test_set.db/img2txts.json $OUTDIR/img2txts.json
cp /scratch/cluster/$USERNAME/vqa_dataset/txt_db/oov_datasets/oov_test_set.db/txt2img.json $OUTDIR/txt2img.json
cp /scratch/cluster/$USERNAME/vqa_dataset/txt_db/oov_datasets/oov_test_set.db/meta.json $OUTDIR/meta.json

