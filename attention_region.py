from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

import sys
sys.path.append('Transformer-Explainability')
from samples.CLS2IDX import CLS2IDX
from baselines.ViT.ViT_LRP import *
from baselines.ViT.ViT_explanation_generator import LRP

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, default='../ILSVRC2012_img_val')
parser.add_argument("--modelname", type=str, default='deit_tiny_patch16_224')

args = parser.parse_args()
print(args)

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

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# initialize ViT pretrained with DeiT
model_name = args.modelname # 'deit_tiny_patch16_224'
model = eval(args.modelname)(pretrained=True).cuda()

print(model)
model.eval()
attribution_generator = LRP(model)

def generate_visualization(original_image, class_index=None):
    transformer_attribution, rollout = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index)
    transformer_attribution = transformer_attribution.detach()
    rollout = rollout.detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    _transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    _transformer_attribution = _transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    _transformer_attribution = (_transformer_attribution - _transformer_attribution.min()) / (_transformer_attribution.max() - _transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, _transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis, transformer_attribution, rollout

if __name__ == '__main__':
    import tqdm
    import os

    image_path = args.datapath
    region_path = 'data/attention_region_' + model_name
    
    label_list = {}
    with open('val_selected.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            label_list[line[0]] = int(line[1])

    total = 0
    count = 0
    
    tqdm_bar = tqdm.tqdm(label_list.keys())
    for i, file in enumerate(tqdm_bar):

        image_file = os.path.join(image_path, file)
        image = Image.open(image_file).convert('RGB').resize((224, 224))
        image_torch = test_transforms(image) 
        image_torch = image_torch.cuda()

        category_index = label_list[file]

        cam, mask, rollout = generate_visualization(image_torch, class_index=category_index)
        
        rollout = rollout.data.cpu().numpy()
        
        os.makedirs(os.path.join(region_path, file.split('/')[0]), exist_ok=True)
        np.save(os.path.join(region_path, file[:-5]+'.npy'), rollout)
        
        total += 1
        output = model(image_torch.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)

        if predicted == label_list[file]:
            count += 1
        tqdm_bar.set_description('%.4f' % (count / total))
                
            