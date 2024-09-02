#!/bin/sh

### Job Params
#TRAIN_SET=configs/data/idd_light.yaml
#TRAIN_SET=configs/data/idd_rainy.yaml
TRAIN_SET=configs/data/idd_bright.yaml
EXP_NAME=yolox608-idd-dark
EPOCHS=1

### Job Body
python train.py \
  -n ${EXP_NAME}-full \
  -d ${TRAIN_SET} \
  --epochs ${EPOCHS} \
  -c configs/model/yolox/yolox_l.yaml | tee joblog_full.out

#python train.py \
#  -n ${EXP_NAME}-e4 \
#  -d ${TRAIN_SET} \
#  --epochs ${EPOCHS} \
#  -c configs/model/ayolo/ayolox_e4.yaml | tee joblog_e4.out

#python train.py \
#  -n ${EXP_NAME}-e3 \
#  -d ${TRAIN_SET} \
#  --epochs ${EPOCHS} \
#  -c configs/model/ayolo/ayolox_e3.yaml | tee joblog_e3.out

#python train.py \
# -n ${EXP_NAME}-e2 \
# -d ${TRAIN_SET} \
# --epochs ${EPOCHS} \
# -c configs/model/ayolo/ayolox_e2.yaml | tee joblog_e2.out

#python train.py \
# -n ${EXP_NAME}-e1 \
# -d ${TRAIN_SET} \
# --epochs ${EPOCHS} \
# -c configs/model/ayolo/ayolox_e1.yaml | tee joblog_e1.out

