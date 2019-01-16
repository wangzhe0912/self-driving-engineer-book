![](/assets/2.jpg)## 霍夫变换在车道线检测中的应用

首先，我们需要对原始图像进行灰度变换、平滑滤波以及Canny边缘检测，得到一副边缘检测后的图像。
然后，我们需要对该图像进行霍夫变换从而找出真实的车道线。

刚才我们已经了解了什么是霍夫变换，我们来看一下如何将霍夫变换用于车道线检测问题。为了实现霍夫变换，我们将会用到OpenCV中`HoughLinesP`的函数，其调用方式如下：
```python
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
```
下面，我们来依次分析相关变量的含义：

1. masked_edges表示的是输入的边缘检测图像（Canny边缘检测的输出结果）。
2. $$\rho$$和$$\theta$$是我们网格在霍夫空间中的距离和角度分辨率。即在霍夫空间中，我们会沿着（θ，ρ）轴分别的网格。其中以像素为单位指定rho，以弧度为单位指定theta。那么$$\rho$$和$$\theta$$的合理取值是什么呢？通常，我们可以设置$$\rho=1$$，$$\theta$$为1度（pi/180）。当时，这些参数可以根据实际情况进行调整~
3. threshold是参与选举的最小曲线交点数。
4. np.array([])仅仅用于占位，我们通常不需要修改。
5. min_line_length表示我们可以接受的直线最小长度。
6. max_line_gap则表示允许将点与直线连接起来的最大距离。
7. lines表示通过霍夫变换后找到的线段列表，其中列表中每个元素都是由(x1,x2,y1,y2)四个坐标点组成的。

对于一副原始图像如下：
![origin](/assets/2.jpg)

一个完整的实现如下所示：
```python
# -*- coding: UTF-8 -*-
"""
# WANGZHE12
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# Read in and grayscale the image
image = mpimg.imread('exit-ramp')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)
ignore_mask_color = 255

# This time we are defining a four sided polygon to mask
imshape = image.shape
# vertices = np.array([[(0, imshape[0]), (0, 0), (imshape[1], 0), (imshape[1], imshape[0])]], dtype=np.int32)

vertices = np.array(
    [
        [
            (0, imshape[0]),
            (imshape[1] / 2 - 50, imshape[0] / 2),
            (imshape[1] / 2 + 50, imshape[0] / 2),
            (imshape[1], imshape[0])
        ]
    ],
    dtype=np.int32
)

cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2  # distance resolution in pixels of the Hough grid
theta = 2 * np.pi/180  # angular resolution in radians of the Hough grid
threshold = 80     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40  #minimum number of pixels making up a line
max_line_gap = 3    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
plt.imshow(lines_edges)
plt.show()
```

处理后得到的结果如下：
![output](/assets/24.jpg)
