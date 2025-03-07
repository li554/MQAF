import argparse
import os
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False
        self.dataset_configs = {
            'CSIQ': {
                'train_ref_path': r'data/IQA/CSIQ/reference_images',
                'train_dis_path': r'data/IQA/CSIQ/distorted_images',
                'val_ref_path': r'data/IQA/CSIQ/reference_images',
                'val_dis_path': r'data/IQA/CSIQ/distorted_images',
                'train_list': r'./datasets/CSIQ/CSIQ_train333.txt',
                'val_list': r'./datasets/CSIQ/CSIQ_test333.txt',
            },
            'KADID': {
                'train_ref_path': r'data/IQA/KADID10K/reference_images',
                'train_dis_path': r'data/IQA/KADID10K/distorted_images',
                'val_ref_path': r'data/IQA/KADID10K/reference_images',
                'val_dis_path': r'data/IQA/KADID10K/distorted_images',
                'train_list': r'./datasets/KADID10K/KADID10K_train333.txt',
                'val_list': r'./datasets/KADID10K/KADID10K_test333.txt',
            },
            'LIVE': {
                'train_ref_path': r'data/IQA/LIVE/reference_images',
                'train_dis_path': r'data/IQA/LIVE/distorted_images',
                'val_ref_path': r'data/IQA/LIVE/reference_images',
                'val_dis_path': r'data/IQA/LIVE/distorted_images',
                'train_list': r'./datasets/LIVE/LIVE_train333.txt',
                'val_list': r'./datasets/LIVE/LIVE_test333.txt',
            },
            'TID2013': {
                'train_ref_path': r'data/IQA/TID_2013/reference_images',
                'train_dis_path': r'data/IQA/TID_2013/distorted_images',
                'val_ref_path': r'data/IQA/TID_2013/reference_images',
                'val_dis_path': r'data/IQA/TID_2013/distorted_images',
                'train_list': r'./datasets/TID_2013/TID_2013_train333.txt',
                'val_list': r'./datasets/TID_2013/TID_2013_test333.txt',
            },
            'CLIVE': {
                'train_ref_path': None,
                'train_dis_path': r'data/IQA/CLIVE/distorted_images',
                'val_ref_path': None,
                'val_dis_path': r'data/IQA/CLIVE/distorted_images',
                'train_list': './datasets/CLIVE/CLIVE_train333.txt',
                'val_list': './datasets/CLIVE/CLIVE_test333.txt',
            },
            'KonIQ': {
                'train_ref_path': None,
                'train_dis_path': r'data/IQA/KonIQ/distorted_images',
                'val_ref_path': None,
                'val_dis_path': r'data/IQA/KonIQ/distorted_images',
                'train_list': './datasets/KonIQ/KonIQ_train333.txt',
                'val_list': './datasets/KonIQ/KonIQ_test333.txt',
            }
        }

    def initialize(self):
        # 添加数据集选择参数
        self._parser.add_argument('--dataset', type=str, default='CSIQ', help='dataset name')
        
        # 基础配置参数
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--name', type=str, default='Memory', help='name of the experiment')
        
        # 设备配置
        self._parser.add_argument('--num_workers', type=int, default=4, help='total workers')
        
        # 模型配置
        self._parser.add_argument('--patch_size', type=int, default=8, help='patch size of Vision Transformer')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--ckpt', type=str, default=None, help='models to be loaded')
        self._parser.add_argument('--seed', type=int, default=42, help='random seed')
        
        # 记忆模型配置
        self._parser.add_argument('--memory_size', type=int, default=1024, help='memory bank size')
        self._parser.add_argument('--train_mode', type=str, default='hybrid', choices=['hybrid', 'memory_only', 'no_memory'], help='training mode')
        self._parser.add_argument('--decov_weight', type=float, default=0.01, help='weight for decorrelation loss')
        self._parser.add_argument('--alpha_weight', type=float, default=0.1, help='weight for sparsity loss')
        self._parser.add_argument('--prob_thresh', type=float, default=0.5, help='probility for mode switch in hybrid mode')
        
        # 数据处理配置
        self._parser.add_argument('--crop_size', type=int, default=224, help='image size')
        self._parser.add_argument('--num_crop', type=int, default=1, help='random crop times')
        self._parser.add_argument('--num_avg_val', type=int, default=5, help='ensemble ways of validation')
        
        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        
        self._opt = self._parser.parse_known_args()[0]
        
        # 加载数据集特定配置
        if self._opt.dataset not in self.dataset_configs:
            raise ValueError(f'Invalid dataset name: {self._opt.dataset}')
            
        config = self.dataset_configs[self._opt.dataset]
        for key, value in config.items():
            setattr(self._opt, key, value)

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("epoch"):
                        load_epoch = max(load_epoch, int(file.split('.')[0].split('_')[1]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("epoch"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

        # set gpu ids
        if len(self._opt.gpu_ids) > 0:
            torch.cuda.set_device(self._opt.gpu_ids[0])

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')