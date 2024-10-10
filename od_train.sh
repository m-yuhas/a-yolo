#!/bin/bash

train_anytime_yolo() {

### Job Params
#TRAIN_SET=configs/data/carla_rain.yaml
#EXP_NAME=yolox608-carla-rain
TRAIN_SET=$1
EXP_NAME=$2
# KITTI EPOCHS=5
# IDD EPOCHS=2
EPOCHS=5
MODELS=("configs/model/yolox/yolox_l.yaml"
        "configs/model/ayolo/ayolox_e4.yaml"
        "configs/model/ayolo/ayolox_e3.yaml"
        "configs/model/ayolo/ayolox_e2.yaml"
)

### Job Body
python train.py \
  -n ${EXP_NAME}-full \
  -d ${TRAIN_SET} \
  --epochs ${EPOCHS} \
  -c "${MODELS[0]}" | tee joblog_full.out

e=4
for model in "${MODELS[@]:1}"
do
  sed -E -i "s/\S*\.pt/${EXP_NAME}-full.pt/g" ${model}
  python train.py \
    -n ${EXP_NAME}-e${e} \
    -d ${TRAIN_SET} \
    --epochs ${EPOCHS} \
    -c ${model} | tee joblog_e${e}.out
  e=$((${e}-1))
done;

}

train_sets=("configs/data/kitti_default.yaml"
	    "configs/data/kitti_bright3.5.yaml"
	    "configs/data/kitti_ice.yaml"
	    "configs/data/kitti_noise.yaml"
	    "configs/data/kitti_rain50.yaml"
	    "configs/data/carla_default.yaml"
	    "configs/data/carla_bright.yaml"
	    "configs/data/carla_ice.yaml"
	    "configs/data/carla_noise.yaml"
	    "configs/data/carla_rain50.yaml"
)
exps=("yolox608-kitti-default"
      "yolox608-kitti-bright"
      "yolox608-kitti-ice"
      "yolox608-kitti-noise"
      "yolox608-kitti-rain"
      "yolox608-carla-default"
      "yolox608-carla-bright"
      "yolox608-carla-ice"
      "yolox608-carla-noise"
      "yolox608-carla-rain"
)

for i in {0..9}; do train_anytime_yolo ${train_sets[${i}]} ${exps[${i}]}; done


