# -*- coding: UTF-8 -*-
"""
Nianshi
无人驾驶工程师 - 计算机视觉 - 一个简单的颜色过滤器
功能：从一副RGB图像中找出白色的车道线
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
color_select = np.copy(image)


red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])

color_select[thresholds] = [0,0,0]

# Display the image
plt.imshow(color_select)
plt.show()
