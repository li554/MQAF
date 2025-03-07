# encoding: utf-8

import logging
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.datasets import IQADataset
from models.memory import MemoryModel
from options.train_options import TrainOptions
from utils.process_image import ToTensor, RandHorizontalFlip, RandCrop, five_point_crop
from utils.util import setup_seed, set_logging
import torch.nn.functional as F


def decov_loss(x):
    x_mean = x - torch.mean(x, dim=1, keepdim=True)
    x_cov = x_mean.mm(x_mean.T)
    loss = torch.norm(x_cov, p='fro') - (torch.diag(x_cov) ** 2 + 1e-6).sum().sqrt()
    return 0.5 * loss


class Train:
    def __init__(self, config):
        self.opt = config
        self.create_model()
        self.init_data()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.opt.learning_rate, 'weight_decay': self.opt.weight_decay}
        ])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.T_max,
                                                                    eta_min=self.opt.eta_min)
        self.train()

    def create_model(self):
        self.model = MemoryModel(num_words=self.opt.memory_size)
        self.model.cuda()

    def init_data(self):
        train_dataset = IQADataset(
            ref_path=self.opt.train_ref_path,
            dis_path=self.opt.train_dis_path,
            txt_file_name=self.opt.train_list,
            transform=transforms.Compose(
                [
                    RandCrop(self.opt.crop_size, self.opt.num_crop),
                    RandHorizontalFlip(),
                    ToTensor(),
                ]
            ),
        )
        val_dataset = IQADataset(
            ref_path=self.opt.val_ref_path,
            dis_path=self.opt.val_dis_path,
            txt_file_name=self.opt.val_list,
            transform=ToTensor(),
        )
        logging.info('number of train scenes: {}'.format(len(train_dataset)))
        logging.info('number of val scenes: {}'.format(len(val_dataset)))

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=False,
            shuffle=False
        )

    def load_model(self):
        models_dir = self.opt.checkpoints_dir
        if os.path.exists(models_dir):
            if self.opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("epoch_"):
                        load_epoch = max(load_epoch, int(file.split('.')[0].split('_')[1]))
                self.opt.load_epoch = load_epoch
                checkpoint = torch.load(os.path.join(models_dir, "epoch_" + str(self.opt.load_epoch) + ".pth"))
                self.model.load_state_dict(checkpoint['model_state_dict'])

                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("epoch_"):
                        found = int(file.split('.')[0].split('_')[1]) == self.opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self.opt.load_epoch
        else:
            assert self.opt.load_epoch < 1, 'Model for epoch %i not found' % self.opt.load_epoch
            self.opt.load_epoch = 0

    def train_epoch(self, epoch):
        losses = []
        self.model.train()
        pred_epoch = []
        labels_epoch = []

        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.train_loader)) as pbar:
            for _, data in enumerate(self.train_loader):
                d_img_org = data['d_img_org'].cuda()
                r_img_org = data['r_img_org'].cuda() if self.opt.train_mode != 'memory_only' else None
                labels = data['score'].view(-1, 1)
                labels = labels.type(torch.FloatTensor).cuda()
                pred, ref_score, dist_score, alpha = self.model(d_img_org, r_img_org)
                # 质量分数回归损失
                score_loss = F.mse_loss(pred, labels)
                # alpha约束
                if self.opt.train_mode == 'hybrid':
                    diff_score_err = torch.abs(dist_score - labels)
                    ref_score_err = torch.abs(ref_score - labels)
                    target_alpha = torch.exp(diff_score_err) / (torch.exp(diff_score_err) + torch.exp(ref_score_err))
                    alpha_loss = F.mse_loss(alpha, target_alpha)
                else:
                    alpha_loss = 0
                # 去相关损失
                dec_loss = decov_loss(self.model.vocab) if self.opt.train_mode != 'no_memory' else 0
                # 组合所有损失
                loss = score_loss + self.opt.decov_weight * dec_loss + self.opt.alpha_weight * alpha_loss
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # save results in one epoch
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = labels.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)

                pbar.update()

        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        ret_loss = np.mean(losses)
        print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))
        logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

        return ret_loss, rho_s, rho_p

    def train(self):
        best_srocc = 0
        best_plcc = 0
        for epoch in range(self.opt.load_epoch, self.opt.n_epoch):
            start_time = time.time()
            logging.info('Running training epoch {}'.format(epoch + 1))
            loss_val, rho_s, rho_p = self.train_epoch(epoch)
            if (epoch + 1) % self.opt.val_freq == 0:
                logging.info('Starting eval...')
                logging.info('Running testing in epoch {}'.format(epoch + 1))
                loss, rho_s, rho_p = self.eval_epoch(epoch)
                logging.info('Eval done...')

                if rho_s + rho_p > best_srocc + best_plcc:
                    best_srocc = rho_s
                    best_plcc = rho_p
                    print('Best now')
                    logging.info('Best now')
                    self.save_model(epoch, "best.pth", loss, rho_s, rho_p)
            logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

    def eval_epoch(self, epoch):
        with torch.no_grad():
            losses = []
            self.model.eval()
            pred_epoch = []
            labels_epoch = []

            with tqdm(desc='Epoch %d - test' % epoch, unit='it', total=len(self.val_loader)) as pbar:
                for _, data in enumerate(self.val_loader):
                    pred = 0
                    for i in range(self.opt.num_avg_val):
                        d_img_org = data['d_img_org'].cuda()
                        r_img_org = data['r_img_org'].cuda()
                        labels = data['score'].view(-1, 1)
                        labels = labels.type(torch.FloatTensor).cuda()
                        d_img_org, r_img_org = five_point_crop(i, d_img=d_img_org, r_img=r_img_org, config=self.opt)
                        r_img_org = r_img_org if self.opt.train_mode != 'memory_only' else None
                        score = self.model(d_img_org, r_img_org)
                        pred += score
                        pbar.update()

                    pred /= self.opt.num_avg_val
                    # compute loss
                    loss = self.criterion(pred, labels)
                    loss_val = loss.item()
                    losses.append(loss_val)

                    # save results in one epoch
                    pred_batch_numpy = pred.data.cpu().numpy()
                    labels_batch_numpy = labels.data.cpu().numpy()
                    pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                    labels_epoch = np.append(labels_epoch, labels_batch_numpy)

            # compute correlation coefficient
            rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            print(
                'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                     rho_p))
            logging.info(
                'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                     rho_p))
            return np.mean(losses), rho_s, rho_p

    def save_model(self, epoch, weights_file_name, loss, rho_s, rho_p):
        print('-------------saving weights---------')
        weights_file = os.path.join(self.opt.checkpoints_dir, weights_file_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch, rho_s, rho_p))


if __name__ == '__main__':
    config = TrainOptions().parse()
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
    setup_seed(config.seed)
    set_logging(config)
    Train(config)
