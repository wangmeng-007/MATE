DATASET_NAME='f30k'
DATA_PATH='/home/s1/ESA-main6/data/f30k'
MODEL_NAME='runs/f30k_butd_region_bert'

cd ../
CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
  --data_path /home/s1/ESA-main6/data/f30k --data_name f30k\
  --logger_name runs/f30k_butd_region_bert/log --model_name runs/f30k_butd_region_bert \
  --num_epochs=20 --lr_update=15 --learning_rate=.0002  --precomp_enc_type basic --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1 --batch_size 128 --hardnum 2

python3 eval.py --dataset f30k --data_path /home/s1/ESA-main6/data/ --model_name runs/f30k_butd_region_bert

