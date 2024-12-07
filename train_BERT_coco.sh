DATASET_NAME='coco'
DATA_PATH='/home/s1/ESA-main1/data/coco'
MODEL_NAME='runs/coco_butd_region_bert'

cd ../
CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
  --data_path /home/s1/ESA-main1/data/coco --data_name coco\
  --logger_name runs/coco_butd_region_bert/log --model_name runs/coco_butd_region_bert \
  --num_epochs=20 --lr_update=15 --learning_rate=.0002 --precomp_enc_type basic --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1 --batch_size 128 --hardnum 2

python3 eval.py --dataset coco  --data_path /home/s1/ESA-main1/data/ --model_name runs/coco_butd_region_bert

