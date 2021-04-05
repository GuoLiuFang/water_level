#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-05-06 11:43
# @Author  : lzneu
# @Site    :
# @File    :

import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from network import East
from common_tools import sigmoid, resize_image
import os
import cv2

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Predict(object):

    def __init__(self, model_path, threshold=0.9, img_size=768):
        self.threshold = threshold
        self.model_path = model_path
        self.lz_size = img_size
        self.model = self.init_model(self.model_path)

    def init_model(self, path):
        east = East()
        model = east.east_network()
        model.load_weights(self.model_path)
        return model

    def load_image(self, img_path, lz_size):
        img = image.load_img(img_path)
        # 修改为强制resize，与训练保持一致
        if False:
            d_wight, d_height = resize_image(img, lz_size)
        else:
            d_wight, d_height = lz_size, lz_size
        # 进一步卷积可能遇到除法向上取整的问题，尽量将输入网络的最大测试图片取为8,16 36,64这样的倍数
        ratio_list = [d_wight / img.width, d_height / img.height]
        wh_tup = (img.width, img.height)
        img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        img = image.img_to_array(img)
        img = preprocess_input(img, mode='tf')
        x = np.expand_dims(img, axis=0)
        return x, wh_tup

    def format_output(self, y, wh_tup, pixel_threshold=0.9):
        y = np.squeeze(y, axis=0)
        y[:, :, :3] = sigmoid(y[:, :, :3])
        score_map =  y[:, :, 0]
        output_score_map = cv2.resize(score_map, wh_tup)
        # output_score_map = cv2.threshold()
        cond = np.where(output_score_map>pixel_threshold, 1, 0)
        return cond

    def infer(self, img_path):
        # load image
        x, wh_tup = self.load_image(img_path, self.lz_size)
        # 开始预测
        y = self.model.predict(x)
        # postprocess
        seg_map = self.format_output(y, wh_tup)

        return seg_map

if __name__ == '__main__':
    model_path = './models/weights_shuiwei_1888T384.005-0.010.h5'
    img_path = './test_image/001_11.jpg'
    model = Predict(model_path, img_size=384)
    seg_map  = model.infer(img_path)
    cv2.imwrite('./tee.jpg', seg_map*255)
    print(seg_map)
