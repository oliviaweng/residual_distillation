#!/bin/bash

layer=$1
en=resnet50_teacher_workers28
batch_size=256
dc=0.0
procedure=RES_NMT
wd=1e-4
dataset=imagenet
tmodel_name=resnet${1}_imagenet
smodel_name=resnet${1}_imagenet_diraconv
aim=${tmodel_name}_${layer}_${procedure}_${en}
echo "teacher_name:"${tmodel_name}
echo "student_name:"${smodel_name}
code_dir=/imagenet-volume/pytorch/python/src/residual_distillation
tboard_dir=/imagenet-volume/imagenet-training/
save_dir=/imagenet-volume/imagenet-training/${aim}/
data_dir=/imagenet-volume/ILSVRC/Data/CLS-LOC/
mode_dir=""
seed=1

python3 ${code_dir}/train_dirac.py --stage RES_NMT \
                       --baseline_epochs 120 \
                       --cutout_length 0 \
                       --procedure ${procedure} \
                       --save_dir ${save_dir} \
                       --smodel_name ${smodel_name} \
                       --tmodel_name ${tmodel_name} \
                       --dataset ${dataset} \
                       --data_dir ${data_dir} \
                       --seed ${seed} \
                       --learning_rate 0.2 \
                       --batch_size ${batch_size} \
                       --aim ${aim} \
                       --start_epoch 0 \
                       --alpha 0.9 \
                       --weight_decay ${wd} \
                       --kd_type margin \
                       --dis_weight 1e-4 \
                       --lr_sch imagenet \
                       --dc ${dc} \
                       --tboard_dir ${tboard_dir} \
                       --experiment_name ${en} \
                       --num_workers 28
                    #    --model_dir ${mode_dir} \
                       
                       
                       
                  
