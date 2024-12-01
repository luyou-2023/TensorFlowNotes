# coding:utf-8
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

# 修改：将 'font.scans-serif' 改为 'font.sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False    # 正常显示正负号

def load_image(path):
    fig = plt.figure("Centre and Resize")
    img = io.imread(path)
    img = img / 255.0  # 归一化图像

    # 原始图像展示
    ax0 = fig.add_subplot(131)
    ax0.set_xlabel(u'Original Picture')
    ax0.imshow(img)

    # 裁剪图像的短边
    short_edge = min(img.shape[:2])
    y = (img.shape[0] - short_edge) / 2
    x = (img.shape[1] - short_edge) / 2
    crop_img = img[int(y):int(y + short_edge), int(x):int(x + short_edge)]  # 裁剪正方形图像

    # 中心裁剪后的图像展示
    ax1 = fig.add_subplot(132)
    ax1.set_xlabel(u"Centre Picture")
    ax1.imshow(crop_img)

    # 将裁剪后的图像缩放到 224x224
    re_img = transform.resize(crop_img, (224, 224))

    # 缩放后的图像展示
    ax2 = fig.add_subplot(133)
    ax2.set_xlabel(u"Resize Picture")
    ax2.imshow(re_img)

    img_ready = re_img.reshape((1, 224, 224, 3))  # 调整图像形状以便于 TensorFlow 处理

    return img_ready

def percent(value):
    return "%.2f" % (value * 100)  # 将数值转换为百分比格式
