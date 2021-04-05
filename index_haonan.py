from flask import Flask, render_template, request, send_from_directory, make_response

import os
import cv2
import json
import shutil
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from network_ import East_
from nms_acc import PredictAfterNms
from common_tools import isImage, sigmoid, clockwise_vertexes, resize_image

import config

plt.switch_backend('Agg')
app = Flask(__name__)

class haonan(object):

    def __init__(self, model_path, threshold=config.pixel_threshold, img_size=640):
        self.threshold = threshold
        self.model_path = model_path
        self.lz_size = img_size
        self.model = self.init_model(self.model_path)

    def init_model(self, path):
        east = East_()
        model = east.east_network()
        model.load_weights(self.model_path)
        return model

    def load_image(self, img_path, lz_size):
        img = image.load_img(img_path)
        # 修改为强制resize，与训练保持一致
        d_wight, d_height = lz_size, lz_size
        # 进一步卷积可能遇到除法向上取整的问题，尽量将输入网络的最大测试图片取为8,16 36,64这样的倍数
        ratio_list = [d_wight / img.width, d_height / img.height]
        img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        img = image.img_to_array(img)
        img = preprocess_input(img, mode='tf')
        x = np.expand_dims(img, axis=0)
        return x, ratio_list

    def format_output(self, y, ratios, out_path, pixel_threshold=0.9):
        y = np.squeeze(y, axis=0)
        y[:, :, :3] = sigmoid(y[:, :, :3])
        cond = np.greater_equal(y[:, :, 0], pixel_threshold)
        activation_pixels = np.where(cond)
        quad_scores, quad_after_nms = predictAfterNms.nms(y, activation_pixels)
        result_list = []
        for score, geo in zip(quad_scores, quad_after_nms):
            if np.amin(score) > 0:
                rescaled_geo = geo / ratios
                # 左上角开始的顺时针
                rescaled_geo_clockwise = clockwise_vertexes(rescaled_geo.reshape((4, 2)))
                rescaled_geo_list = np.reshape(rescaled_geo_clockwise, (8,)).tolist()
                point_0 = (rescaled_geo_list[0], rescaled_geo_list[1])
                point_1 = (rescaled_geo_list[2], rescaled_geo_list[3])
                point_2 = (rescaled_geo_list[4], rescaled_geo_list[5])
                point_3 = (rescaled_geo_list[6], rescaled_geo_list[7])
                result_list.append([point_0, point_1, point_2, point_3])
        return result_list

    def infer(self, in_path, visualize=False):
        img_path_list = []
        if os.path.isfile(in_path):
            img_path = in_path
            img_path_list = [img_path]
            res_dir = os.path.dirname(in_path)
        elif os.path.isdir(in_path):
            res_dir = in_path
            for file_name in os.listdir(in_path):
                if isImage(file_name):
                    img_path = os.path.join(in_path, file_name)
                    img_path_list.append(img_path)
        for img_path in img_path_list:
            print("CHARACTER DETECTING: ", img_path)
            file_name = os.path.basename(img_path)
            txt_path = os.path.join(res_dir, file_name.replace(file_name.split('.')[-1], '') + 'txt')

            # load image
            x, ratios = self.load_image(img_path, self.lz_size)
            # 开始预测
            y = self.model.predict(x)
            # postprocess
            return self.format_output(y, ratios, txt_path, pixel_threshold=self.threshold)

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
predictAfterNms = PredictAfterNms()
_haonan = haonan('./models/weights_shuiwei_1888T640.014-0.038.h5')
graph = tf.get_default_graph()

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['POST'])
def parse():
    # 清除较早的暂存文件
    auto_remove(root)

    f = request.files['file']
    _str = str(datetime.datetime.now())
    time_code = _str.replace("-","").replace(":","").replace(".","").replace(" ","")
    path = root+time_code+"/"
    img_path = path+f.filename
    output_path = path+"output.png"
    os.mkdir(path)
    f.save(img_path)

    with graph.as_default():
        quads = _haonan.infer(img_path)

    plt.figure()
    plt.imshow(cv2.imread(img_path, 1)[...,::-1])
    for quad in quads:
        for i in range(4):
            plt.plot([quad[i-1][0], quad[i][0]], [quad[i-1][1], quad[i][1]], c="C2")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    return send_from_directory(path, "output.png")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug = True)
