# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2019/11/22
#   description:
#
#================================================================
import os
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import torch
import random
import json


class A3DMILDataset(Dataset):

    def __init__(self, root_dir, batch_size, phase='train', transforms=None):

        self.normal_vids_dir = os.path.join(root_dir, 'normal')
        self.anomaly_vids_dir = os.path.join(root_dir, 'abnormal')
        self.normal_vids = os.listdir(self.normal_vids_dir)[:-1]
        self.anomaly_vids = os.listdir(self.anomaly_vids_dir)
        self.all_vids = {'normal': self.normal_vids, 'abnormal': self.anomaly_vids}

        self.num_anomaly = len(self.anomaly_vids)
        self.num_normal = len(self.normal_vids)

        self.batch_size = batch_size    # 1 abnormal and 1 normal
        self.n_exp = self.batch_size / 2

        self.root_dir = root_dir

        self.phase = phase

        self.abnormal_dict = json.load(
            open(
                '/u/xiziwang/projects/stad-cvpr2020/baselines/VideoActionRecognition/pytorch-i3d/A3D_2.0.json'
            ))

    def __getitem__(self, index):

        All_features = []
        name = []
        if self.phase == 'train':
            label = 'normal'
            name.append(self.all_vids[label][index])
            # index = np.random.randint(0, self.num_normal)
            dire = os.path.join(os.path.join(self.root_dir, label), self.all_vids[label][index])
            normal_clip_num = 0
            for i, iv in enumerate(os.listdir(dire)):
                if i % 8 == 0:
                    feature = np.load(os.path.join(dire, iv))
                    All_features.append(torch.from_numpy(feature))
                    normal_clip_num += 1
        label = 'abnormal'
        # index = np.random.randint(0, self.num_anomaly)
        dire = os.path.join(os.path.join(self.root_dir, label), self.all_vids[label][index])
        abnormal_clip_num = 0
        name.append(self.all_vids[label][index])
        for i, iv in enumerate(os.listdir(dire)):
            if i % 8 == 0:
                feature = np.load(os.path.join(dire, iv))
                All_features.append(torch.from_numpy(feature))
                abnormal_clip_num += 1

        if self.phase == 'train':
            Normal_labels = np.zeros(normal_clip_num, dtype='uint8')
            Anomaly_labels = np.ones(abnormal_clip_num, dtype='uint8')
            All_labels = np.concatenate((Normal_labels, Anomaly_labels))

        else:
            start = self.abnormal_dict[self.all_vids[label][index]]['anomaly_start']
            if start == None:
                start = 0
            end = self.abnormal_dict[self.all_vids[label][index]]['anomaly_end']
            Anomaly_labels = np.zeros(abnormal_clip_num, dtype='uint8')
            for i in range(abnormal_clip_num):
                if i * 8 >= start and i * 8 < end:
                    Anomaly_labels[i] = 1.0
            All_labels = Anomaly_labels
        All_features = torch.stack(All_features, dim=0)
        All_labels = torch.from_numpy(All_labels)
        # name = self.all_vids[label][index]
        return name, All_features, All_labels

    def __len__(self):
        return self.num_anomaly

    def callback(self):
        self.anomaly_vids = np.random.permutation(self.anomaly_vids)
        self.normal_vids = np.random.permutation(self.normal_vids)
