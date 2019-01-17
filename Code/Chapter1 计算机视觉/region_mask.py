# -*- coding: UTF-8 -*-
"""
Nianshi
无人驾驶工程师 - 计算机视觉 - 感兴趣区域提取
功能：从一副RGB图像中找出车道线的感兴趣区域
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


image = mpimg.imread('attachment/rgb_image.jpg')
print('This image is: ', type(image), 'with dimensions:', image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
# Note: always make a copy rather than simply using "="
region_select = np.copy(image)

left_bottom = [0, 430]
right_bottom = [760, 430]
apex = [380, 240]

# 多项式拟合函数，用于拟合y=Ax+B表达式中的A和B
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)


XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY < (XX * fit_left[0] + fit_left[1])) | (YY < (XX * fit_right[0] + fit_right[1])) | (YY > (XX * fit_bottom[0] + fit_bottom[1]))

region_select[region_thresholds] = [0, 0, 0]

# Display the image
plt.imshow(region_select)

plt.show()
