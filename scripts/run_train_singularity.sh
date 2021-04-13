CONFIG=$1

cd uniter

python train_vqa.py --config config/$CONFIG
