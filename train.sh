#!/bin/bash

# CSIQ数据集 - 混合模式训练
python train.py --dataset CSIQ --train_mode hybrid --n_epoch 200 --learning_rate 1e-4 --name CSIQ_hybrid
# CSIQ数据集 - 纯记忆模式训练
python train.py --dataset CSIQ --train_mode memory_only --n_epoch 200 --learning_rate 1e-4 --name CSIQ_memory_only
# CSIQ数据集 - 无记忆模式训练
python train.py --dataset CSIQ --train_mode no_memory --n_epoch 200 --learning_rate 1e-4 --name CSIQ_no_memory

# TID2013数据集 - 混合模式训练
python train.py --dataset TID2013 --train_mode hybrid --n_epoch 200 --learning_rate 8e-5 --name TID2013_hybrid
# TID2013数据集 - 纯记忆模式训练
python train.py --dataset TID2013 --train_mode memory_only --n_epoch 200 --learning_rate 8e-5 --name TID2013_memory_only
# TID2013数据集 - 无记忆模式训练
python train.py --dataset TID2013 --train_mode no_memory --n_epoch 200 --learning_rate 8e-5 --name TID2013_no_memory

# KADID10K数据集 - 混合模式训练
python train.py --dataset KADID --train_mode hybrid --n_epoch 200 --learning_rate 8e-5 --name KADID_hybrid
# KADID10K数据集 - 纯记忆模式训练
python train.py --dataset KADID --train_mode memory_only --n_epoch 200 --learning_rate 8e-5 --name KADID_memory_only
# KADID10K数据集 - 无记忆模式训练
python train.py --dataset KADID --train_mode no_memory --n_epoch 200 --learning_rate 8e-5 --name KADID_no_memory
