#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/15 10:46 上午
# @Author  : lzneu
# @Site    : 
# @File    : config.py
"""
adeast配置文件
"""

# 必要的配置文件
epsilon = 1e-4
num_channels = 3
feature_layers_range = range(5, 1, -1)
feature_layers_num = len(feature_layers_range)
pixel_size = 2 ** feature_layers_range[-1]
locked_layers = False
side_vertex_pixel_threshold = 0.9
pixel_threshold = 0.9
trunc_threshold = 0.1