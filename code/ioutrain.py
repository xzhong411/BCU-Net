import glob

import torch
import torch.nn as nn

import sklearn.metrics as metrics
from core.models import UNet
import os
import argparse
import time

from torch.autograd import Variable

from core.unet_parts import RecallCrossEntropy
from core.utils import calculate_Accuracy, get_model, get_data
from pylab import *
import random
from test import fast_test
from warmlearnrate import adjust_learning_rate

plt.switch_backend('agg')

# --------------------------------------------------------------------------------
model_name= 'BCU-Net'
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--epochs', type=int, default=50,
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

parser.add_argument('--data_path', type=str, default='../data/CVC-ClinicDB',help='dir of the all img')
parser.add_argument('--model_save', type=str, default='../models/clin_best_model.pth',help='dir of the model.pth')

# parser.add_argument('--data_path', type=str, default='../data/DRHAGIS',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/drha_best_model.pth',help='dir of the model.pth')

# parser.add_argument('--data_path', type=str, default='../data/Kvasir-SEG',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/SEG_best_model.pth',help='dir of the model.pth')

#parser.add_argument('--data_path', type=str, default='../data/DRIVE_1',help='dir of the all img')
#parser.add_argument('--model_save', type=str, default='../models/dri_best_model.pth',help='dir of the model.pth')
#
# parser.add_argument('--data_path', type=str, default='../data/STARE_1',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/stare_best_model.pth',help='dir of the model.pth')
parser.add_argument('--my_description', type=str, default='',
                    help='some description define your training')
parser.add_argument('--batch_size', type=int, default=2,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=512,
                    help='the training img size')
# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='0,1,2,3',
                    help='the gpu used')

args = parser.parse_args()
print(args)
# --------------------------------------------------------------------------------


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

model = BCU-Net(n_channels=3,n_classes=args.n_class)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
if args.use_gpu:
    model.cuda()
    print('GPUs used: (%s)' % args.gpu_avaiable)
    print('------- success use GPU --------')
print("  ''''''''")
EPS = 1e-12
# define path
data_path = args.data_path

train_img_list=glob.glob(os.path.join(data_path, 'train/image/*.tif'))
test_img_list=glob.glob(os.path.join(data_path, 'test/image/*.tif'))


# img_list = get_img_list(args.data_path, flag='training')
# test_img_list = get_img_list(args.data_path, flag='test')


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
criterion = RecallCrossEntropy()
softmax_2d = nn.Softmax2d()

IOU_best = 0

print ('This model is %s_%s_%s_%s' % (model_name, args.n_class, args.img_size,args.my_description))
if not os.path.exists(r'../models/%s_%s' % (model_name, args.my_description)):
    os.mkdir(r'../models/%s_%s' % (model_name, args.my_description))

with open(r'../logs/%s_%s.txt' % (model_name, args.my_description), 'w+') as f:
    f.write('This model is %s_%s: ' % (model_name, args.my_description)+'\n')
    f.write('args: '+str(args)+'\n')
    f.write('training lens: '+str(len(train_img_list))+' | test lens: '+str(len(test_img_list)))
    f.write('\n\n---------------------------------------------\n\n')



for epoch in range(args.epochs):
    model.train()

    begin_time = time.time()
    print ('This model is %s_%s_%s_%s' % (
        model_name, args.n_class, args.img_size, args.my_description))
    random.shuffle(train_img_list)

    if 'arg' in args.data_path:
        if (epoch % 10 ==  0) and epoch != 0 and epoch < 400:
            args.lr /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # optimizer = torch.optim.Adam(model.parameters(),lr=adjust_learning_rate(optimizer, epoch, args.lr, args.epochs//8, args.epochs, 0.9))
    for i, (start, end) in enumerate(zip(range(0, len(train_img_list), args.batch_size),
                                         range(args.batch_size, len(train_img_list) + args.batch_size,
                                               args.batch_size))):
        path = train_img_list[start:end]
        img, gt, tmp_gt, img_shape,label_ori = get_data(args.data_path, path, img_size=args.img_size, gpu=args.use_gpu)
        optimizer.zero_grad()


        out1,out2,out = model(img)
        out = torch.log(softmax_2d(out) + EPS)
        loss = criterion(out, gt)
        loss = loss + criterion(torch.log(softmax_2d(out1) + EPS), gt)
        loss = loss + criterion(torch.log(softmax_2d(out2) + EPS), gt)
        out = torch.log(softmax_2d(out) + EPS)
        loss.backward()
        optimizer.step()

        ppi = np.argmax(out.cpu().data.numpy(), 1)
        tmp_out = ppi.reshape([-1])
        tmp_gt = tmp_gt.reshape([-1])

        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
        miou, mdice, Acc, Se, Sp, IU, f1 = calculate_Accuracy(my_confusion)



        print(str('model: {:s}_{:s} | epoch_batch: {:d}_{:d} | loss: {:f}  | miou: {:.3f} | mdice: {:.3f} ').format(model_name, args.my_description,epoch, i, loss.item(), miou,mdice))

    print('training finish, time: %.1f s' % (time.time() - begin_time))

    if epoch % 10 == 0 and epoch != 0:
        torch.save(model.state_dict(), args.model_save)
        print('success save Nucleus_best model')