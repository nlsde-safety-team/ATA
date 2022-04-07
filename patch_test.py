from unittest.mock import patch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
from torchvision import transforms
import timm
import argparse
import csv
import os
import numpy as np
import matplotlib
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import tqdm
import json
from PIL import Image
import torchvision.transforms.functional as TF



parser = argparse.ArgumentParser()

parser.add_argument("--datapath", type=str, default='../ILSVRC2012_img_val')
parser.add_argument('--modelname', type=str, default='deit_tiny_patch16_224')
parser.add_argument('--patchlog', type=str, default = 'logs/attack-deit_tiny_patch16_224+position-PlogPCol_Embed+patchsize-32+loss-logP+')
parser.add_argument('--norm', type=int, default = 1)
args = parser.parse_args()


PATCHSIZE = 32
DATA_PATH = args.datapath
BATCH_SIZE = 10
NORMALIZE = (args.norm == 1)
print(NORMALIZE)


log_dir = args.patchlog

def save_image(image_tensor, save_file):
    image_tensor = image_tensor.clone()
    image_tensor = image_tensor[:16]
    # print(image_tensor.shape)
    if not NORMALIZE:
        image_tensor[:, 0, :, :] = image_tensor[:, 0, :, :] * 0.5 + 0.5
        image_tensor[:, 1, :, :] = image_tensor[:, 1, :, :] * 0.5 + 0.5
        image_tensor[:, 2, :, :] = image_tensor[:, 2, :, :] * 0.5 + 0.5
    else:
        image_tensor[:, 0, :, :] = image_tensor[:, 0, :, :] * 0.229 + 0.485 
        image_tensor[:, 1, :, :] = image_tensor[:, 1, :, :] * 0.224 + 0.456 
        image_tensor[:, 2, :, :] = image_tensor[:, 2, :, :] * 0.225 + 0.406 
    torchvision.utils.save_image(image_tensor, save_file, nrow=4)

if args.modelname == 'resnet50':
    model = models.resnet50(pretrained=True).cuda()
elif args.modelname == 'densenet121':
    model = models.densenet121(pretrained=True).cuda()
else:
    model = timm.create_model(args.modelname, pretrained=True).cuda()

model.eval()


class MyDataset(Dataset):
    def __init__(self, data_path, patch_path=None, mask_path=None):
        self.data_path = data_path
        self.patch_path = patch_path
        self.mask_path = mask_path
        
        
        if not NORMALIZE:
            self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            self.normalize,
        ])


        self.datas = []
        with open('val_selected.txt', 'r') as f:
            for line in f.readlines():
                file, label = line.split()[:2]
                label = int(label)
                self.datas.append((file, label))
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        image_file, label = self.datas[index]

        image = Image.open(os.path.join(self.data_path, image_file)).convert('RGB').resize((224, 224))
        image = self.transform(image)
        
        if self.patch_path is None:
            patch = torch.randn((3, PATCHSIZE, PATCHSIZE))
        else:
            patch = np.load(os.path.join(self.patch_path, image_file[:-5]+'.npy'))
            patch = torch.from_numpy(patch)

            patch = self.normalize(patch)

        mask = np.load(os.path.join(self.mask_path, image_file[:-5]+'.npy'))
        
        return image, label, patch, mask, image_file
    
mydataset = MyDataset(DATA_PATH, os.path.join(args.patchlog, 'patch_trained'), os.path.join(args.patchlog, 'masks'))
data_loader = DataLoader(mydataset, batch_size=BATCH_SIZE, num_workers=2)

with torch.no_grad():
    test_total, test_actual_total, test_success, test_acc = 0, 0, 0, 0
    tqdm_bar = tqdm.tqdm(data_loader)
    for i, data in enumerate(tqdm_bar):
        images, labels, patchs, masks, files = data
        batchsize = labels.shape[0]
        test_total += batchsize
        
        patchs_pad = patchs * masks
        masks_pad = masks
            
        images = images.cuda()
        labels = labels.cuda()
        patchs_pad = patchs_pad.cuda()
        masks_pad = masks_pad.cuda()
        
        inputs = images * (1 - masks_pad) + patchs_pad * masks_pad
        
        if not NORMALIZE:
            inputs.clamp_(-1, 1)
        else:
            inputs = TF.normalize(inputs, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            inputs.clamp_(0, 1)
            inputs = TF.normalize(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        outputs = model(images)
        _, predicted_clean = torch.max(outputs.data, 1)
        probs_clean = torch.nn.functional.softmax(outputs, dim=1)
        
        outputs = model(inputs)
        _, predicted_adv = torch.max(outputs.data, 1)
        probs_adv = torch.nn.functional.softmax(outputs, dim=1)
        
        for _id in range(batchsize):
            if predicted_clean[_id] == labels[_id]:
                test_actual_total += 1
                if predicted_adv[_id] != labels[_id]:
                    test_success += 1
            if predicted_adv[_id] == labels[_id]:
                    test_acc += 1
            
            
            with open(os.path.join(log_dir, 'test_'+args.modelname+'.txt'), 'a') as f:
                f.write('%s, %s, clean_prob, %.6f, adv_prob, %.6f\n' % (files[_id], args.modelname, probs_clean[_id][labels[_id]], probs_adv[_id][labels[_id]]))
            
        
        tqdm_bar.set_description("ass:%.4f, clean_acc:%.4f, adv_acc:%.4f" % (test_success / test_actual_total, test_actual_total/test_total, (test_acc / test_total)))
        
        if i % 100 == 0:
            save_image(inputs, os.path.join(log_dir, 'test_inputs.png'))
with open(os.path.join(log_dir, 'test.txt'), 'a') as f:
    f.write('%s, ass,%.6f, clean_acc,%.6f, adv_acc,%.6f\n' % (args.modelname, test_success / test_actual_total, test_actual_total/test_total, (test_acc / test_total)))
            
    