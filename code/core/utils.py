from __future__ import division
import numpy as np
import os

from PIL import Image
import cv2
import csv
import torch
#from models import *
import pickle as pkl

from torch import nn
from torch.autograd import Variable

# from core.models import UNet
from core.models import  UNet


def get_data(data_path, img_path, img_size=256, gpu=True,flag=False):  #img_path='../data/Medical_Datasets/train/image/im0077.png'

    def get_label(label):
        tmp_gt = label.copy()
        label = label.astype(np.int64)
        label = Variable(torch.from_numpy(label)).long()
        if gpu:
            label = label.cuda()
        return label,tmp_gt

    images = []
    labels = []
    tmp_gts = []

    img_shape =[]
    label_ori = []

    batch_size = len(img_path)
    for i in range(batch_size):
        # img_path = os.path.join(data_path, 'train/image/', img_name[i])
        # label_path = os.path.join(data_path, 'label/image/', img_name[i])

        img = cv2.imread(img_path[i])
        label = cv2.imread(img_path[i].replace('image','label').replace('jpg','png'),0)
        img_shape.append(img.shape)
        label_ori.append(label)
        # label[label < 150] = 0
        # label[label > 150] = 1
        label[label <128] = 0
        label[label>=128] = 1
        img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()

        if gpu:
            img = img.cuda()

        label, tmp_gt = get_label(label)
        images.append(img)
        labels.append(label)
        tmp_gts.append(tmp_gt)

    images = torch.stack(images)
    labels = torch.stack(labels)
    tmp_gts = np.stack(tmp_gts)


    label_ori = np.stack(label_ori)

    return images, labels, tmp_gts, img_shape,label_ori


def calculate_Accuracy(confusion):
    confusion=np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
    res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    f1 = 2 * confusion[1][1] / (2 * confusion[1][1] + confusion[1][0] + confusion[0][1])
    IU = tp / (pos + res - tp)
    dice = 2 * tp / (pos+res)
    meanDice = np.mean(dice)
    meanIU = np.mean(IU)
    Acc = np.sum(tp) / np.sum(confusion)
    Se = confusion[1][1] / (confusion[1][1]+confusion[0][1])
    Sp = confusion[0][0] / (confusion[0][0]+confusion[1][0])
    # Fpr = confusion[1][0]
    # Tpr = confusion[1][1]

    # return  meanIU,meanDice,Acc,Se,Sp,IU,f1
    return  IU[1],dice[1],Acc,Se,Sp,IU,f1

def get_model(model_name):
    if model_name=='UNet':
        return UNet


def update_variance(self, labels, pred1, pred2):
            criterion = nn.CrossEntropyLoss(weight = self.class_weight, ignore_index=255, reduction = 'none')
            kl_distance = nn.KLDivLoss( reduction = 'none')
            loss = criterion(pred1, labels)

            #n, h, w = labels.shape
            #labels_onehot = torch.zeros(n, self.num_classes, h, w)
            #labels_onehot = labels_onehot.cuda()
            #labels_onehot.scatter_(1, labels.view(n,1,h,w), 1)

            variance = torch.sum(kl_distance(self.log_sm(pred1),self.sm(pred2)), dim=1)
            exp_variance = torch.exp(-variance)
            #variance = torch.log( 1 + (torch.mean((pred1-pred2)**2, dim=1)))
            #torch.mean( kl_distance(self.log_sm(pred1),pred2), dim=1) + 1e-6
            print(variance.shape)
            print('variance mean: %.4f'%torch.mean(exp_variance[:]))
            print('variance min: %.4f'%torch.min(exp_variance[:]))
            print('variance max: %.4f'%torch.max(exp_variance[:]))
            #loss = torch.mean(loss/variance) + torch.mean(variance)
            loss = torch.mean(loss*exp_variance) + torch.mean(variance)
            return loss

