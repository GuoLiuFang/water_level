# -*-coding: utf-8 -*-
# 使用Pil进行图像增强
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from IPython.display import display
import random


def resize_blur(img):
    """
    ① NEAREST：最近滤波。从输入图像中选取最近的像素作为输出像素。
    ② BILINEAR：双线性内插滤波。在输入图像的2*2矩阵上进行线性插值。
    ③ BICUBIC：双立方滤波。在输入图像的4*4矩阵上进行立方插值。
    ④ ANTIALIAS：平滑滤波。对所有可以影响输出像素的输入像素进行高质量的重采样滤波，以计算输出像素值。
    """
    seed = random.random()*3+1  # 放缩倍数
    filter_type = random.choice([0, 1, 2, 3])  # 插值方式
    width, height = img.size[0], img.size[0]
    if filter_type == 0:
        img = img.resize((int(width/seed), int(height/seed)),Image.NEAREST) 
        img = img.resize((width, height),Image.NEAREST) 
    elif filter_type == 1:
        img = img.resize((int(width/seed), int(height/seed)),Image.BILINEAR) 
        img = img.resize((width, height),Image.BILINEAR) 
    elif filter_type == 2:
        img = img.resize((int(width/seed), int(height/seed)),Image.BICUBIC) 
        img = img.resize((width, height),Image.BICUBIC) 
    else:
        img = img.resize((int(width/seed), int(height/seed)),Image.ANTIALIAS) 
        img = img.resize((width, height),Image.ANTIALIAS) 
    return img

def direct_blur(img):
    return img.filter(ImageFilter.BLUR)


def guass_blur(img):
    radius = random.random()*3  # 0-3
    return img.filter(ImageFilter.GaussianBlur(radius=radius))  

def apply_random_blur(img):
    blur_type = random.choice([0,0,0,0,0,0,0,2,3,1])   # 0-原图
    if blur_type == 1:
        img = resize_blur(img)
    elif blur_type == 2:
        img = guass_blur(img)
    elif blur_type == 3:
        img = direct_blur(img)
    return img, blur_type

if __name__ == '__main__':
    for i in range(20):
        print(random.choice([0,10,0,0,1,3,4,5]))
