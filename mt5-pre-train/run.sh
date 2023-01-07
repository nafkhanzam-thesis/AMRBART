#! /bin/bash

export CUDA_VISIBLE_DEVICES=0
dataset="amrbart"
MODEL=$1

datapath="../../datasets/$dataset"
interval=1

lr=5e-5

outpath=../outputs/${dataset}-$(echo $MODEL | sed -r 's/\//-/g')-Unifiedtextinf-JointDenoise-6task-${lr}-AMREOS

mkdir -p $outpath
echo "OutputDir: $outpath"

python -u run_multitask_unified_pretraining.py \
  --train_file $datapath/train.jsonl \
  --val_file $datapath/val.jsonl \
  --test_file $datapath/test.jsonl \
  --output_dir $outpath \
  --mlm \
  --mlm_amr \
  --mlm_text \
  --mlm_amr_plus_text \
  --mlm_text_plus_amr \
  --mlm_joint_to_amr \
  --mlm_joint_to_text \
  --block_size 512 \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --model_type $MODEL \
  --model_name_or_path $MODEL \
  --save_total_limit 2 \
  --do_train \
  --do_eval \
  --evaluate_during_training  \
  --num_train_epochs 100  \
  --learning_rate $lr \
  --joint_train_interval $interval \
  --warmup_steps 2500 \
  --max_steps 100000 \
  --logging_steps 100 \
  --fp16 \
  --no_cache \
  --overwrite_output_dir 2>&1 | tee $outpath/run.log
