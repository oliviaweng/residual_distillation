layer=$1
dc=$2
procedure=$3
wd=$4
dataset=imagenet
tmodel_name=resnet${1}_imagenet
smodel_name=resnet${1}_imagenet_diraconv
aim=${smodel_name}_${layer}_${procedure}
echo "teacher_name:"${tmodel_name}
echo "student_name:"${smodel_name}
save_dir=/imagenet-volume/code/${aim}/
data_dir=/imagenet-volume/ILSVRC/Data/CLS-LOC/
mode_dir=""
seed=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_dirac.py --stage RES_NMT \
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
                       --batch_size 512 \
                       --aim ${aim} \
                       --start_epoch 0 \
                       --alpha 0.9 \
                       --weight_decay ${wd} \
                       --kd_type margin \
                       --dis_weight 1e-4 \
                       --lr_sch imagenet \
                       --dc ${dc} \
                       --model ${mode_dir} \
                       
                       
                       
                  
