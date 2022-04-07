from statistics import median_grouped
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
import torch.nn.functional as F
from functools import reduce
import random
import skimage
import math

random.seed(2333)

parser = argparse.ArgumentParser()

parser.add_argument("--datapath", type=str, default='../ILSVRC2012_img_val')
parser.add_argument('--modelname', type=str, default='deit_tiny_patch16_224')
parser.add_argument("--attnpath", type=str, default='data/attention_region_deit_tiny_patch16_224')
parser.add_argument("--embedpath", type=str, default='data/embed_position_deit_tiny_patch16_224')

args = parser.parse_args()

PATCHSIZE = 32
DATA_PATH = args.datapath
BATCH_SIZE = 8
MAX_ITER = 250
LR = 1.0 
ATTN_PATH = args.attnpath
EMBED_PATH = args.embedpath

NORMALIZE = True
if 'vit' == args.modelname[:3]:
    NORMALIZE = False

def make_log_dir():
    logs = {
        'attack': args.modelname,  
        'position': 'PlogPCol_Embed',
        'patchsize': PATCHSIZE,
        'loss': 'logP',
    }
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    dir_name = 'logs/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    return dir_name

log_dir = make_log_dir()

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

model = timm.create_model(args.modelname, pretrained=True).cuda()
model.eval()

with open(os.path.join(log_dir, 'model.txt'), 'w') as f:
    f.write('%s\n' % str(model))

        
def make_mask_embed(shape, num_pixel, embed):
    mask = torch.zeros(shape)
    embed_reshape = embed.reshape((256))
    sort_arg = np.argsort(-embed_reshape)
    for i in range(int(num_pixel)):
        x = sort_arg[i] // 16
        y = sort_arg[i] % 16
        assert embed_reshape[sort_arg[i]] == embed[x, y]
        mask[x, y] = 1
    return mask
    

class MyDataset(Dataset):
    def __init__(self, data_path, patch_path=None):
        self.data_path = data_path
        self.patch_path = patch_path
        
        if not NORMALIZE:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        image_np = np.array(image, dtype=np.float32)
        image = self.transform(image)
        
        embed_image = np.load(os.path.join(EMBED_PATH, image_file[:-5]+'.npy'))
        
        rollout  = np.load(os.path.join(ATTN_PATH, image_file[:-5]+'.npy'))
        grad_token = np.zeros((196))
        cls_sum = rollout[0,0,1:].sum()
        for i in range(1,197):
          rollout[0,0,i] /= cls_sum
          for j in range(1,197):
              if i != j:
                grad_token[i-1] -= rollout[0,j,i]*np.log2(rollout[0,j,i])

        grad_token = np.reshape(grad_token,(14,14))
        grad_token /= grad_token.sum()
        grad_token = np.floor(grad_token* 1024)
        
        
        for i in range(14):
            for j in range(14):
                while grad_token[i, j] > 255:
                    grad_token[i, j] -= 1
        
        for i in range(1024,int(grad_token.sum())):
            x = random.randint(0, 13)
            y = random.randint(0, 13)
            while grad_token[x, y] <= 100:
                x = random.randint(0, 13)
                y = random.randint(0, 13)
            grad_token[x, y] -= 1
        
        for i in range(int(grad_token.sum()), 1024):
            x = random.randint(0, 13)
            y = random.randint(0, 13)
            while grad_token[x, y] >= 250:
                x = random.randint(0, 13)
                y = random.randint(0, 13)
            grad_token[x, y] += 1


        patch = torch.randn_like(image)
        total = 0.
        count = 0
        mask = torch.zeros_like(image)
        for i in range(0, 224, 16):
            for j in range(0, 224, 16):
                total += (32*32) / (14*14)
                diff = (total - count) // 1 + 1

                _mask = make_mask_embed((16, 16), grad_token[i//16, j//16], embed_image[i:i+16, j:j+16])
                
                mask[:, i:i+16, j:j+16] = _mask
                
                count += diff
        
        assert mask.sum() == 3 * 32 * 32
        
        return image, label, patch, mask, image_file
    

mydataset = MyDataset(DATA_PATH)
data_loader = DataLoader(mydataset, batch_size=BATCH_SIZE, num_workers=2)

tqdm_bar = tqdm.tqdm(data_loader)
for data in tqdm_bar:
    images, labels, patchs, masks, files = data
    batchsize = labels.shape[0]
    
    patchs_pad = patchs * masks
    masks_pad = masks
    
    save_image(patchs_pad, os.path.join(log_dir, 'init_patchs_pad.png'))
    save_image(masks[:1], os.path.join(log_dir, 'mask.png'))
    
    images = images.cuda()
    labels = labels.cuda()
    patchs_pad = patchs_pad.cuda()
    masks_pad = masks_pad.cuda()
        
    patchs_pad.requires_grad_(True)
    
    optimizer = torch.optim.Adam([patchs_pad], lr=LR)# , weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150], gamma=1/3)
    
    clean_attn = []
    attention_list = clean_attn
    model(images)
    
    
    
    for iter in tqdm.tqdm(range(MAX_ITER)):
        # print(torch.min(images), torch.max(images))
        inputs = images * (1 - masks_pad) + patchs_pad * masks_pad
        
        if not NORMALIZE:
            inputs.clamp_(-1, 1)
        else:
            inputs = TF.normalize(inputs, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            inputs.clamp_(0, 1)
            inputs = TF.normalize(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        adv_attn = []
        attention_list = adv_attn
        outputs = model(inputs)
        
        
        
        # print(outputs.shape)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        probs = probs.index_select(1, labels)
        # print(probs.shape)
        
        loss_logP =  (-torch.log(1-probs+1e-10) * torch.eye(batchsize).cuda()).sum() / batchsize
        probs = (probs * torch.eye(batchsize).cuda()).sum() / batchsize

        loss = loss_logP
        optimizer.zero_grad()
        
        loss.backward(retain_graph=True)
        
        optimizer.step()
        scheduler.step()
        
        if not NORMALIZE:
            patchs_pad.data.clamp_(-1, 1)
        else:
            patchs_pad.data = TF.normalize(patchs_pad.data, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            patchs_pad.data.clamp_(0, 1)
            patchs_pad.data = TF.normalize(patchs_pad.data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        tqdm_bar.set_description("loss:%.4f, logP:%.4f, prob:%.4f" % (loss, loss_logP, probs))
        
        if iter % 50 == 0:
            save_image(patchs_pad, os.path.join(log_dir, 'patchs_pad.png'))
            save_image(inputs, os.path.join(log_dir, 'inputs.png'))
    
    with torch.no_grad():
        outputs = model(images)
        probs_clean = torch.nn.functional.softmax(outputs, dim=1)
        inputs = images * (1 - masks_pad) + patchs_pad * masks_pad
        if not NORMALIZE:
            inputs.clamp_(-1, 1)
        else:
            inputs = TF.normalize(inputs, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            inputs.clamp_(0, 1)
            inputs = TF.normalize(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        outputs = model(inputs)
        probs_adv = torch.nn.functional.softmax(outputs, dim=1)
    for i in range(batchsize):
        with open(os.path.join(log_dir, 'train.txt'), 'a') as f:
            f.write('%s, %s, clean_prob, %.6f, adv_prob, %.6f\n' % (files[i], args.modelname, probs_clean[i][labels[i]], probs_adv[i][labels[i]]))
    
    
    
    # save_patch
    patchs_trained = patchs_pad 
    
    for i in range(batchsize):
        _patchs = patchs_trained[i].data.cpu()
        _mask = masks[i].data.cpu()
        
        os.makedirs(os.path.join(log_dir, 'patch_trained', files[i].split('/')[0]), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'masks', files[i].split('/')[0]), exist_ok=True)

        if not NORMALIZE:
            _patchs = _patchs * 0.5 + 0.5
        else:
            _patchs = TF.normalize(_patchs, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        
        np.save(os.path.join(log_dir, 'patch_trained', files[i][:-5] + '.npy'), _patchs.numpy())
        torchvision.utils.save_image(_patchs, os.path.join(log_dir, 'patch_trained', files[i][:-5] + '.png'))

        np.save(os.path.join(log_dir, 'masks', files[i][:-5] + '.npy'),_mask.numpy())