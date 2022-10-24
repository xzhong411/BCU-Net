# -*- coding: utf-8 -*-
import glob

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import xlrd
import xlwt
import xlsxwriter
import sklearn.metrics as metrics

from xlrd import open_workbook
from xlutils.copy import copy
import os
import argparse
import time

from core.models import UNet
from core.utils import calculate_Accuracy, get_data, get_model
from pylab import *

plt.switch_backend('agg')

# --------------------------------------------------------------------------------

model_name = 'BCU-Net'

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')
# ---------------------------
# params do not need to change
# ---------------------------
parser.add_argument('--epochs', type=int, default=250,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--lr', type=float, default=0.0015,
                    help='initial learning rate')
parser.add_argument('--GroupNorm', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='decide to use the BatchNorm')
# ---------------------------
# model
# ---------------------------
#parser.add_argument('--data_path', type=str, default='../data/CHASEDB1_1',help='dir of the all img')
#parser.add_argument('--model_save', type=str, default='../models/chase_best_model.pth',help='dir of the model.pth')
# parser.add_argument('--data_path', type=str, default='../data/CVC-ClinicDB',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/clin_best_model.pth',help='dir of the model.pth')

parser.add_argument('--data_path', type=str, default='../data/Kvasir-SEG',help='dir of the all img')
parser.add_argument('--model_save', type=str, default='../models/unet_conext_reacll_loc-glo/SEG_best_model_1.pth',help='dir of the model.pth')

# parser.add_argument('--data_path', type=str, default='../data/Kvasir-SEG',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/SEG_best_model.pth',help='dir of the model.pth')

#parser.add_argument('--data_path', type=str, default='../data/DRIVE_1',help='dir of the all img')
#parser.add_argument('--model_save', type=str, default='../models/dri_best_model.pth',help='dir of the model.pth')
#
# parser.add_argument('--data_path', type=str, default='../data/STARE_1',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/stare_best_model.pth',help='dir of the model.pth')
parser.add_argument('--my_description', type=str, default='',
                    help='some description define your training')
parser.add_argument('--best_model', type=str,  default='final.pth',
                    help='the pretrain model')
parser.add_argument('--batch_size', type=int, default=2,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=512,
                    help='the train img size')

# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='0,1,2,3',
                    help='the gpu used')

args = parser.parse_args()

def fast_test(model, args, model_name):
    softmax_2d = nn.Softmax2d()
    EPS = 1e-12
    MIOU = []
    MDICE = []

    data_path = args.data_path
    test_img_list = glob.glob(os.path.join(data_path, 'test/image/*.png'))
    for i, img_path in enumerate(test_img_list):
        start = time.time()
        save_res_path = (img_path.replace('test/image', 'testsave'))
        # img_path = test_img_list[start:end]
        img, gt, tmp_gt, img_shape,label_ori = get_data(args.data_path, [img_path], img_size=args.img_size, gpu=args.use_gpu)
        model.eval()


        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            print(img_path)
            model = nn.DataParallel(model)
        with torch.no_grad():
            out1,out2,out= model(img)
            pred = np.array(out.data.cpu()[0])
            save = np.zeros(shape=(512, 512))
            save[pred[0] < 0.5] = 255
            save[pred[1] < 0.5] = 0
            cv2.imwrite(save_res_path, save)

            out = torch.log(softmax_2d(out) + EPS)

            out = F.upsample(out, size=(img_shape[0][0], img_shape[0][1]), mode='bilinear')
            out = out.cpu().data.numpy()

            y_pred = out[:, 1, :, :]
            y_pred = y_pred.reshape([-1])
            ppi = np.argmax(out, 1)
            tmp_out = ppi.reshape([-1])

            tmp_gt = label_ori.reshape([-1])

            my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
            miou,mdice,Acc,Se,Sp,IU,f1 = calculate_Accuracy(my_confusion)
            MDICE.append(mdice)
            MIOU.append(miou)


            end = time.time()
            print(str(i + 1) + r'/' + str(
                len(test_img_list)) + ': ' + '| miou: {:.3f} | mdice: {:.3f} '.format(miou, mdice) + '  |  time:%s' % (end - start))

    print('miou: %s  |  mdice: %s ' % (
        str(np.mean(np.stack(MIOU))), str(np.mean(np.stack(MDICE)))))

    # todo create excel file
    xl = xlsxwriter.Workbook(r'../result.xls')

    # todo add to sheet
    sheet = xl.add_worksheet('chasedb1')

    # todo add datas to cell
    sheet.write(12, 1, str(np.mean(np.stack(MIOU))))
    sheet.write(12, 2, str(np.mean(np.stack(MDICE))))

    # todo close the file
    xl.close()

    # store test information
    with open(r'../logs/%s_%s.txt' % (model_name, args.my_description), 'a+') as f:
        f.write('miou: %s  |  mdice: %s ' % (
            str(np.mean(np.stack(MIOU))), str(np.mean(np.stack(MDICE)))))
        f.write('\n\n')

    return np.mean(np.stack(MIOU))


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

    model = BCU-Net(n_channels=3, n_classes=args.n_class)
    model = nn.DataParallel(model)
    if args.use_gpu:
        model.cuda()
    if True:
        model.load_state_dict(torch.load(args.model_save))
        print('success load models: %s_%s' % (model_name, args.my_description))

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    print('This model is %s_%s_%s' % (model_name, args.n_class, args.img_size))
    fast_test(model, args, model_name)


