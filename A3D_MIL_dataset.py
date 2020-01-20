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

        self.num_clips_per_sample = 16

        self.abnormal_dict_train = json.load(
            open(
                '/u/xiziwang/projects/stad-cvpr2020/baselines/VideoActionRecognition/pytorch-i3d/A3D_2.0_train.json'
            ))
        self.abnormal_dict_val = json.load(
            open(
                '/u/xiziwang/projects/stad-cvpr2020/baselines/VideoActionRecognition/pytorch-i3d/A3D_2.0_val.json'
            ))
        self.abnormal_dict_train.update(self.abnormal_dict_val)
        self.abnormal_dict = self.abnormal_dict_train
        self.normal_dict = json.load(
            open(
                '/u/xiziwang/projects/stad-cvpr2020/baselines/VideoActionRecognition/pytorch-i3d/A3D_2.0_train.json'
            ))

    def __getitem__(self, index):

        if self.phase == 'train':

            normal_sample = self.get_normal_sample_nonoverlap(index)
            abnormal_sample = self.get_abnormal_sample_non_overlap(index)
            All_features = torch.cat([normal_sample['feature'], abnormal_sample['feature']], dim=0)

            All_labels = torch.from_numpy(
                np.concatenate((normal_sample['labels'], abnormal_sample['labels'])))

        else:
            abnormal_sample = self.get_abnormal_sample_non_overlap(index)
            All_features = abnormal_sample['feature']
            All_labels = torch.from_numpy(abnormal_sample['labels'])

        return All_features, All_labels
        # dire = os.path.join(os.path.join(self.root_dir, label), self.all_vids[label][index])
        # abnormal_clip_num = 0
        # name.append(self.all_vids[label][index])
        # for i, iv in enumerate(os.listdir(dire)):
        # if i % 8 == 0:
        # feature = np.load(os.path.join(dire, iv))
        # All_features.append(torch.from_numpy(feature))
        # abnormal_clip_num += 1

        # if self.phase == 'train':
        # Normal_labels = np.zeros(normal_clip_num, dtype='uint8')
        # Anomaly_labels = np.ones(abnormal_clip_num, dtype='uint8')
        # All_labels = np.concatenate((Normal_labels, Anomaly_labels))

        # else:
        # start = self.abnormal_dict[self.all_vids[label][index]]['anomaly_start']
        # if start == None:
        # start = 0
        # end = self.abnormal_dict[self.all_vids[label][index]]['anomaly_end']
        # Anomaly_labels = np.zeros(abnormal_clip_num, dtype='uint8')
        # for i in range(abnormal_clip_num):
        # if i * 8 >= start and i * 8 < end:
        # Anomaly_labels[i] = 1.0
        # All_labels = Anomaly_labels
        # All_features = torch.stack(All_features, dim=0)
        # All_labels = torch.from_numpy(All_labels)
        # # name = self.all_vids[label][index]
        # return name, All_features, All_labels

    def get_normal_sample_nonoverlap(self, index):

        dire = os.path.join(os.path.join(self.root_dir, 'normal'), self.all_vids['normal'][index])
        normal_clip_num = 0

        features = []
        clips = {}
        normal_clip_num = int(normal_clip_num)
        last_feat = 0
        for i, iv in enumerate(os.listdir(dire)):
            clip_start = int(iv.split('_')[0])
            clips[int(clip_start)] = torch.from_numpy(np.load(os.path.join(dire, iv)))
            normal_clip_num += 1
            # if clip_start == len(os.listdir(dire)) - 1:
            # last_feat = torch.from_numpy(np.load(os.path.join(dire, iv)))
        # clips[normal_clip_num] = last_feat
        # normal_clip_num += 1
        num_frames = self.normal_dict[self.all_vids['normal'][index]]['num_frames']
        for k in range(num_frames):
            if k in clips.keys():
                l_feat = clips[k]
            else:
                clips[k] = l_feat
        normal_clip_num = len(clips.keys())
        pad_num = 0
        if normal_clip_num % self.num_clips_per_sample == 0:
            pad_num = 0
        else:
            pad_num = self.num_clips_per_sample - (normal_clip_num % self.num_clips_per_sample)
        for i in range(pad_num):
            clips[normal_clip_num + i] = clips[normal_clip_num - 1]

            # if type(clips[k]) == int:
            # print(dire)
            # print(clips)

        normal_clip_num = len(clips.keys())
        avg_num = normal_clip_num / self.num_clips_per_sample
        avg_num = int(avg_num)
        # print(self.all_vids['normal'][index], normal_clip_num, avg_num)

        for i in range(self.num_clips_per_sample):
            avg_feat = []
            for j in range(avg_num):
                avg_feat.append(clips[i * avg_num + j])
            avg_feat = torch.stack(avg_feat, dim=0)
            avg = torch.mean(avg_feat, dim=0)
            features.append(avg)
        # print('features')
        # print(features)

        normal_labels = np.zeros(len(features))
        if len(features) > 0:
            features = torch.stack(features, dim=0)
        else:
            features = torch.tensor([])
        return {"feature": features, "clip_num": len(features), "labels": normal_labels}

    def get_abnormal_sample_non_overlap(self, index):
        dire = os.path.join(os.path.join(self.root_dir, 'abnormal'),
                            self.all_vids['abnormal'][index])
        abnormal_clip_num = 0
        # name.append(self.all_vids[label][index])
        features = []
        clips = {}
        abnormal_clip_num = int(abnormal_clip_num)
        last_feat = 0
        for i, iv in enumerate(os.listdir(dire)):
            clip_start = int(iv.split('_')[0])
            # if clip_start % 16 == 0:
            clips[int(clip_start)] = torch.from_numpy(np.load(os.path.join(dire, iv)))
            abnormal_clip_num += 1
            # if clip_start == len(os.listdir(dire)) - 1:
            # last_feat = torch.from_numpy(np.load(os.path.join(dire, iv)))
        # clips[abnormal_clip_num] = last_feat
        # abnormal_clip_num += 1
        num_frames = self.abnormal_dict[self.all_vids['abnormal'][index]]['num_frames']
        for k in range(num_frames):
            if k in clips.keys():
                l_feat = clips[k]
            else:
                clips[k] = l_feat
        abnormal_clip_num = len(clips.keys())
        if abnormal_clip_num % self.num_clips_per_sample == 0:
            pad_num = 0
        else:
            pad_num = self.num_clips_per_sample - (abnormal_clip_num % self.num_clips_per_sample)
        for i in range(pad_num):
            clips[abnormal_clip_num + i] = clips[abnormal_clip_num - 1]
        abnormal_clip_num = len(clips.keys())
        avg_num = abnormal_clip_num / self.num_clips_per_sample
        avg_num = int(avg_num)

        for i in range(self.num_clips_per_sample):
            avg_feat = []
            for j in range(avg_num):
                avg_feat.append(clips[i * avg_num + j])
            avg_feat = torch.stack(avg_feat, dim=0)
            avg = torch.mean(avg_feat, dim=0)
            features.append(avg)

        if self.phase == 'train':

            Anomaly_labels = np.ones(len(features))
        else:
            start = self.abnormal_dict[self.all_vids['abnormal'][index]]['anomaly_start']
            if start == None:
                start = 0
            end = self.abnormal_dict[self.all_vids['abnormal'][index]]['anomaly_end']
            Anomaly_labels = np.zeros(len(features))
            for i in range(len(features)):
                feat_start = i * avg_num
                feat_end = (i + 1) * avg_num
                feat_middle = (feat_start + feat_end) / 2
                if feat_middle >= start and feat_middle < end:
                    Anomaly_labels[i] = 1.0
                # elif feat_end >= start and feat_end < end:
                # Anomaly_labels[i] = 1.0
        return {
            "feature": torch.stack(features, dim=0),
            'clip_num': len(features),
            'labels': Anomaly_labels
        }

    def get_normal_sample(self, index):
        # name.append(self.all_vids['normal'][index])
        # index = np.random.randint(0, self.num_normal)
        dire = os.path.join(os.path.join(self.root_dir, 'normal'), self.all_vids['normal'][index])
        normal_clip_num = 0
        features = []
        for i, iv in enumerate(os.listdir(dire)):
            if i % 8 == 0:
                feature = np.load(os.path.join(dire, iv))
                features.append(torch.from_numpy(feature))
                normal_clip_num += 1

        Normal_labels = np.zeros(normal_clip_num)
        if len(features) > 0:
            features = torch.stack(features, dim=0)
        else:
            features = torch.tensor([])
        return {"feature": features, 'clip_num': normal_clip_num, 'labels': Normal_labels}

    def get_abnormal_sample(self, index):
        dire = os.path.join(os.path.join(self.root_dir, 'abnormal'),
                            self.all_vids['abnormal'][index])
        abnormal_clip_num = 0
        # name.append(self.all_vids[label][index])
        features = []
        for i, iv in enumerate(os.listdir(dire)):
            if i % 8 == 0:
                feature = np.load(os.path.join(dire, iv))
                features.append(torch.from_numpy(feature))
                abnormal_clip_num += 1

        if self.phase == 'train':

            Anomaly_labels = np.ones(abnormal_clip_num)
        else:
            start = self.abnormal_dict[self.all_vids['abnormal'][index]]['anomaly_start']
            if start == None:
                start = 0
            end = self.abnormal_dict[self.all_vids['abnormal'][index]]['anomaly_end']
            Anomaly_labels = np.zeros(abnormal_clip_num)
            for i in range(abnormal_clip_num):
                if i * 8 >= start and i * 8 < end:
                    Anomaly_labels[i] = 1.0
        return {
            "feature": torch.stack(features, dim=0),
            'clip_num': abnormal_clip_num,
            'labels': Anomaly_labels
        }

    def __len__(self):
        if self.phase == 'train':
            return self.num_normal
        else:
            return self.num_anomaly

    def callback(self):
        self.anomaly_vids = np.random.permutation(self.anomaly_vids)
        self.normal_vids = np.random.permutation(self.normal_vids)
