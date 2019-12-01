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

class A3DMILDataset(Dataset):
    def __init__(self, root_dir, batch_size, transforms = None):

        self.normal_vids_dir = os.path.join(root_dir, 'normal')
        self.anomaly_vids_dir = os.path.join(root_dir, 'anomaly')

        self.normal_vids = os.listdir(self.normal_vids_dir)  
        self.anomaly_vids = os.listdir(self.anomaly_vids_dir)  
        self.all_vids = {'normal': self.normal_vids, 'anomaly':self.anomaly_vids}

        self.num_anomaly = len(self.anomaly_vids)
        self.num_normal = len(self.normal_vids) 

        self.batch_size = batch_size # 1 anomaly and 1 normal
        self.n_exp = self.batch_size/2

        self.root_dir = root_dir

    def __getitem__(self, index):

        label = 'normal' 
        # Normal_features = []
        # Anomaly_features = []
        dire = os.path.join(os.path.join(self.root_dir, label), self.all_vids[label][index]  ) 
        normal_clip_num = len(os.listdir(dire) )
        # Normal_features = np.zeros(shape=(normal_clip_num, 1024) ) 
        All_features = []
        for i,iv in enumerate(os.listdir(dire)):
            feature = np.load(os.path.join(dire, iv) ) 
            All_features.append(torch.from_numpy(feature) ) 
        label = 'anomaly'
        # abnormal_clip_num = 0
        dire = os.path.join(os.path.join(self.root_dir, label), self.all_vids[label][index]  ) 
        abnormal_clip_num = len(os.listdir(dire) )
        # Anomaly_features = np.zeros(shape = ( abnormal_clip_num, 1024) ) 
        for iv in os.listdir(dire):
            feature = np.load(os.path.join(dire, iv) ) 
            All_features.append(torch.from_numpy(feature) )  

        Normal_labels = np.zeros(normal_clip_num, dtype = 'uint8') 

        Anomaly_labels = np.ones(abnormal_clip_num, dtype = 'uint8') 
        
        All_features = torch.stack(All_features, dim=0) 
        All_labels = np.concatenate((Normal_labels, Anomaly_labels)) 
        # print(All_labels.shape) 
        All_labels = torch.from_numpy(All_labels) 
        return All_features, All_labels
         
    def __len__(self):
        return self.num_anomaly
