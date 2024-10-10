#!/bin/bash



test_anytime_yolo() {
### Job Params
#TEST_SET1=configs/data/carla_default.yaml
#TEST_SET2=configs/data/carla_noise.yaml
#MODEL1=yolox608-carla-default
#MODEL2=yolox608-carla-noise
TEST_SET1=$1
TEST_SET2=$2
MODEL1=$3
MODEL2=$4


e=5
while [[ ${e} -ge 2 ]];
do
  model=${MODEL1}
  if [[ ${e} -eq 5 ]]; then
    model=${model}-full.pt
  else
    model=${model}-e${e}.pt
  fi
  echo -e "Model \033[0;32m ${model} \033[0m tested on \033[0;33m ${TEST_SET1} \033[0m @exit ${e}"
  python train.py -c configs/model/yolox/yolox_l.yaml -d ${TEST_SET1} --ckpt ${model} --test 2>/dev/null | grep -A 1 "mAP" | grep "Batch 0,"
  echo -e "Model \033[0;32m ${model} \033[0m tested on \033[0;33m ${TEST_SET2} \033[0m @exit ${e}"
  python train.py -c configs/model/yolox/yolox_l.yaml -d ${TEST_SET2} --ckpt ${model} --test 2>/dev/null | grep -A 1 "mAP" | grep "Batch 0,"

  model=${MODEL2}
  if [[ ${e} -eq 5 ]]; then
    model=${model}-full.pt
  else
    model=${model}-e${e}.pt
  fi
  echo -e "Model \033[0;32m ${model} \033[0m tested on \033[0;33m ${TEST_SET1} \033[0m @exit ${e}"
  python train.py -c configs/model/yolox/yolox_l.yaml -d ${TEST_SET1} --ckpt ${model} --test 2>/dev/null | grep "Batch 0,"
  echo -e "Model \033[0;32m ${model} \033[0m tested on \033[0;33m ${TEST_SET2} \033[0m @exit ${e}"
  python train.py -c configs/model/yolox/yolox_l.yaml -d ${TEST_SET2} --ckpt ${model} --test 2>/dev/null | grep "Batch 0,"

  e=$((${e}-1))
done;
}

test_set1=("configs/data/kitti_bright3.5.yaml"
           "configs/data/kitti_ice.yaml"
           "configs/data/kitti_noise.yaml"
           "configs/data/kitti_rain50.yaml"
           "configs/data/carla_bright.yaml"
           "configs/data/carla_ice.yaml"
           "configs/data/carla_noise.yaml"
           "configs/data/carla_rain50.yaml"
)
test_set2=("configs/data/kitti_default.yaml"
	   "configs/data/kitti_default.yaml"
	   "configs/data/kitti_default.yaml"
	   "configs/data/kitti_default.yaml"
	   "configs/data/carla_default.yaml"
	   "configs/data/carla_default.yaml"
           "configs/data/carla_default.yaml"
	   "configs/data/carla_default.yaml"
)
model1=("yolox608-kitti-bright"
	"yolox608-kitti-ice"
	"yolox608-kitti-noise"
	"yolox608-kitti-rain"
	"yolox608-carla-bright"
	"yolox608-carla-ice"
	"yolox608-carla-noise"
	"yolox608-carla-rain"
)
model2=("yolox608-kitti-default"
	"yolox608-kitti-default"
	"yolox608-kitti-default"
	"yolox608-kitti-default"
	"yolox608-carla-default"
	"yolox608-carla-default"
	"yolox608-carla-default"
	"yolox608-carla-default"
)

for i in {0..7}; do
  echo -e "\033[0;31m Model ${model1[${i}]} \033[0m"
  test_anytime_yolo ${test_set1[${i}]} ${test_set2[${i}]} ${model1[${i}]} ${model2[${i}]};
done
