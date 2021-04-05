# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import imageio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from model.CPD_ResNet_models import CPD_ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CPD_Predict(object):
    def __init__(self, test_size, model_path):
        self.test_size = test_size
        self.model = CPD_ResNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.transform = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    def _preprocess(self, image_path):
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        image_shape = np.asarray(image).shape
        image = self.transform(image).unsqueeze(0)
        return image, image_shape
        
    def _predict(self, image):
        image = image.to(device)
        _, dets = self.model(image)
        return dets

    def _postprocess(self, image, dets, image_shape):
        res = F.upsample(dets, size=image_shape[:2], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        return res

    def __call__(self, image_path):
        image, image_shape = self._preprocess(image_path)
        dets = self._predict(image)
        res = self._postprocess(image, dets, image_shape)
        return res


if __name__ == '__main__':
    test_size = 352
    model_path = './models/CPD_Resnet/CPD_300.pth'
    image_path = '/LegendStart/CPD/datasets/shuiwei/'
    save_path = '/LegendStart/CPD/datasets/bingling/visualization/'


    cpd_pre = CPD_Predict(test_size, model_path)
    for file in os.listdir(image_path):
        if file.endswith('jpg') or file.endswith('jpeg') \
            or file.endswith('png'):
            image_file_path = os.path.join(image_path, file)
            save_file_path = os.path.join(save_path, 'vis_' + file)

            res = cpd_pre(image_file_path)
            imageio.imwrite(save_file_path, res)


