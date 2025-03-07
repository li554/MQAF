#!/bin/bash

# 全参考对比实验（200轮次）

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

# 无参考对比实验（200轮次）

# KonIQ数据集 - 纯记忆模式训练
python train.py --dataset KonIQ --train_mode memory_only --n_epoch 200 --learning_rate 8e-5 --name KonIQ_memory_only
# CLIVE数据集 - 纯记忆模式训练
python train.py --dataset CLIVE --train_mode memory_only --n_epoch 200 --learning_rate 8e-5 --name CLIVE_memory_only

# 消融实验（200轮次）
# 记忆单元数量分析 - TID2013数据集
python train.py --dataset TID2013 --train_mode hybrid --memory_size 128 --n_epoch 200 --learning_rate 8e-5 --name TID2013_memory_size_128
python train.py --dataset TID2013 --train_mode hybrid --memory_size 512 --n_epoch 200 --learning_rate 8e-5 --name TID2013_memory_size_512
python train.py --dataset TID2013 --train_mode hybrid --memory_size 1024 --n_epoch 200 --learning_rate 8e-5 --name TID2013_memory_size_1024

# 去相关损失分析 - TID2013数据集
python train.py --dataset TID2013 --train_mode hybrid --decov_weight 0 --n_epoch 200 --learning_rate 8e-5 --name TID2013_decov_0
python train.py --dataset TID2013 --train_mode hybrid --decov_weight 0.001 --n_epoch 200 --learning_rate 8e-5 --name TID2013_decov_0.001
python train.py --dataset TID2013 --train_mode hybrid --decov_weight 0.01 --n_epoch 200 --learning_rate 8e-5 --name TID2013_decov_0.01