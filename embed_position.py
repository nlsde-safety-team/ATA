import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
import timm
import tqdm
import os
import torchvision
from torch.utils.data import DataLoader
import json


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, default='../ILSVRC2012_img_val')
parser.add_argument("--modelname", type=str, default='deit_tiny_patch16_224')

args = parser.parse_args()
print(args)

PATCHSIZE=16
model_name = args.modelname 
model = timm.create_model(model_name, pretrained=True).cuda()
model.eval()

with open(os.path.join('model.txt'), 'w') as f:
    f.write('%s\n' % str(model))

image_path = args.datapath 
output_path = f'./data/embed_position_{args.modelname}'

# Setup the transformation

if 'vit' == args.modelname[:3]:
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
else:
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

print(test_transforms)

data_files = []
with open('val_selected.txt', 'r') as f:
    for line in f.readlines():
        line = line.split()
        data_files.append(line[0])

for file in tqdm.tqdm(data_files):
    image_file = os.path.join(image_path, file)
    image = Image.open(image_file).convert('RGB').resize((224, 224))
    image_torch = test_transforms(image).unsqueeze(0)
    image_torch = image_torch.cuda()
    
    origin_embed = model.patch_embed(image_torch)
    
    embed_mask_2 = torch.zeros((224, 224))
    
    for i in range(16):
        for j in range(16):
            new_image_torch = image_torch.clone()
            for x in range(14):
                for y in range(14):
                    new_image_torch[:, :, x*16+i, y*16+j] = 0
                
            new_embed = model.patch_embed(new_image_torch)
            
            diff = origin_embed - new_embed

            diff = (diff ** 2) ** 0.5
            
            for x in range(14):
                for y in range(14):
                    embed_mask_2[x*16+i, y*16+j] = diff[:, x*14+y, :].sum().data.cpu()
            
    
    embed_mask_2 = embed_mask_2.data.cpu().numpy()
    embed_mask_2 -= np.min(embed_mask_2)
    embed_mask_2 /= np.max(embed_mask_2)
    
    os.makedirs(os.path.join(output_path, file.split('/')[0]), exist_ok=True)
    np.save(os.path.join(output_path, file[:-5] + '.npy'), embed_mask_2)
    
