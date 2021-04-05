#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/25 5:12 下午
# @Author  : lzneu
# @Site    : 
# @File    : common_tools.py

import numpy as np
import re
import os
import shutil
import cv2
from PIL import Image
class DrawLines(object):

    # 画框工具类
    def __init__(self):
        pass

    def draw_img(self, img_path, boxes, color='r', scores=None):
        """
        support 3colors，保存到输出路径
        :param img:图片路径
        :param boxes:list[list]一张图片的所有框坐标
        :param color:支持三种颜色rgb
        :return: 画图后的图片
        """
        img = Image.open(img_path)
        img = np.array(img.convert('RGB'))
        img = img.copy()
        if color == 'r' or color == 'red':
            color = (255, 0, 0)  # red
        elif color == 'g' or color == 'green':
            color = (0, 255, 0)  # green
        else:
            color = (0, 0, 255)
        for i in range(len(boxes)):
            box = boxes[i]
            if len(box) == 4:
                tmp = []
                for i in box:
                    tmp += [i[0], i[1]]
            box = tmp
            # if np.linalg.norm(box[2] - box[0]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            #     continue  # 过滤宽或高小于5个像素点的值
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[6]), int(box[7])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
            if scores is not None:
                score = scores[i]
                img = cv2.putText(img, str(score), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200,100,100), 1)
        return img





def isImage(name):
    """
    通过文件后缀判断是否是一个图片名称
    Args:
        name: 文件名

    Returns: bool 是否是图片名称

    """
    return name.split('.')[-1] in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def clockwise_vertexes(xy_list):
    """
    返回顺时针时针的坐标
    Args:
        xy_list: 四个点坐标，顺序不限

    Returns: 修正后的坐标顺序，左上角开始的顺时针顺序

    """
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x, 注意不能 换成 x+y 最小的点作为第一个点！！
    # if two has same x, choose that with smallest y,
    # 按列排序并返回原索引 xy已经不对应了
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    # 取第一个点 x最小的点
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    # 斜率list
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
               / (xy_list[index, 0] - xy_list[first_v, 0] + 1e-4)
    # 中间大的斜率 就是右下角的点
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    # 中间点的截距b。。
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    # 通过y=kx+b来求出y大于mid的点 就是第二个点
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
            xy_list[second_v, 0] - xy_list[fourth_v, 0] + 1e-4)
    # 注意这个斜率和数学直角坐标系反的   。。。
    # 保证左上是第一个点
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    # 逆时针转顺时针
    reorder_xy_list[[1, 3], :] = reorder_xy_list[[3, 1], :]

    return reorder_xy_list


def resize_image(im, max_img_size=896):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


s_dic = {'，': ',', '（': '(', '）': ')', '；': ';',
         '：': ':', '‘': '\'', '’': '\'', '“': '\"',
         '”': '\"'}


def replace_points(s):
    """
    字符串统一转换，中文替换为英文
    Args:
        s: 原始字符串
    Returns:
        统一后的字符串
    """
    for k, v in s_dic.items():
        s = s.replace(k, v)
    return s


def patstr(string, pattern_str):
    rule = re.compile(r'[' + pattern_str + ']+')
    tmp_list = rule.findall(string)
    res = ''.join(tmp_list)
    return res


def get_files(path='/Users/lz/work_data/'):
    all = []
    for fpathe, dirs, fs in os.walk(path):  # os.walk获取所有的目录
        for f in fs:
            filename = os.path.join(fpathe, f)
            all.append(filename)
    return all


if __name__ == '__main__':
    pattern_str = r"\/—\-()\"\'（）、‘’“”。，,.\a-zA-Z0-9\u4e00-\u9fa5"
    print(patstr('dfasdf---_——-//a\"\'afa4323"()\"\'（）‘’“”""k', pattern_str))
