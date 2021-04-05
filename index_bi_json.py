# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import cv2
import time
import json
import shutil
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from lib_CPD.CPD_ResNet_models import CPD_ResNet

from flask import Flask, render_template, request, send_from_directory, make_response

device = torch.device('cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
app = Flask(__name__)

class CPD_Predict(object):
    def __init__(self, test_size, model_path):
        self.test_size = test_size
        self.model = CPD_ResNet()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
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

def auto_remove(root):
    dt_now = datetime.datetime.now()
    print(dt_now)
    folders = [x for x in os.listdir(root) if x[0]!='.']
    for folder in folders:
        dt = datetime.datetime(int(folder[0:4]),int(folder[4:6]),int(folder[6:8]),int(folder[8:10]),int(folder[10:12]),int(folder[12:14]))
        if (dt_now-dt)>datetime.timedelta(0,300):
            shutil.rmtree(root+folder)
    return

root = "/tmp/"
test_size = 352
model_path = './models/CPD_300.pth'
cpd_pre = CPD_Predict(test_size, model_path)

@app.route('/', methods = ['POST'])
def parse():
    start_time = time.time()

    f = request.files['file']
    _str = str(datetime.datetime.now())
    time_code = _str.replace("-","").replace(":","").replace(".","").replace(" ","")
    path = root+time_code+"/"
    img_path = path+f.filename
    output_path = path+"output.png"
    os.mkdir(path)
    f.save(img_path)

    mask = cpd_pre(img_path)

    shutil.rmtree(path)
    serverTime = time.time()-start_time

    _output = {"code": 0, "msg": "OK", "serverTime": serverTime, "data":mask.tolist()}
    _output_json = json.dumps(_output, ensure_ascii=False).encode('utf8')
    response = make_response(_output_json)
    response.headers["Content-Type"] = "application/json"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7501, debug=True)
