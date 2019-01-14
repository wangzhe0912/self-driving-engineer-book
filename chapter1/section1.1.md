# 1.1 计算机基础

为了让车辆能够自主的行驶，我们首先需要让车辆能够正常感知到车辆周围的环境信息。
在我们人类驾驶的过程中，我们是通过眼睛来观察车道线的位置，从而判断在哪里转弯，如何正常行驶等。而在自动驾驶的汽车中，我们需要使用摄像头和一些其他类型的传感器来实现类似的功能。我们将要解决的第一个问题就是如何去识别车道线。


下图是一个车载摄像头拍摄到的图片，观察这个图片，你觉得哪些特征可以帮助我们来定位车道线呢？
![前置摄像头](/assets/2.jpg)

事实上，我们可以从以下四个方面来帮助我们定位车道线：

1. 颜色
2. 形状
3. 方向
4. 位置









## 结合感兴趣区域提取与颜色过滤器提取车道线
上面的过程中，我们已经学习了如何分别使用颜色过滤器和感兴趣区域提取对图像处理，下面，我们来看下如何结合两种方式来更加精确的检测出车道线：
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('attachment/rgb_image.jpg')

ysize = image.shape[0]
xsize = image.shape[1]
color_select= np.copy(image)
line_image = np.copy(image)

# Define our color criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

left_bottom = [0, 430]
right_bottom = [760, 430]
apex = [380, 240]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | (image[:,:,1] < rgb_threshold[1]) | (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & (YY > (XX*fit_right[0] + fit_right[1])) & (YY < (XX*fit_bottom[0] + fit_bottom[1]))
# Mask color selection
color_select[color_thresholds] = [0, 0, 0]
# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# Display our two output images
plt.imshow(color_select)
plt.imshow(line_image)

# uncomment if plot does not display
plt.show()
```
通过上述代码，我们可以得到处理后的图像如下：
![结合感兴趣区域提取与颜色过滤器提取车道线](/assets/8.jpeg)

目前为止，我们已经可以找到简单图像的车道线了，但实际上，目前的算法距离真实的应用还有很大的差距。事实上，车道线并不总是白色的，还有黄色的；此外，在不同的条件下（白天、夜晚、雨天等等）即使是相同颜色的车道线也无法通过我们简单的颜色过滤算法检测到。

接下来，我们需要通过更加复杂的计算机视觉方法来检测任何有颜色的线条。

## 计算机视觉
既然我们已经提到了计算机视觉，那么计算机视觉究竟是什么呢？
计算机视觉实际就是通过利用各种算法，使得计算机能够向我们人类一样看到一个有颜色、形状的世界。计算机视觉是一个很大的领域，在本书中，我们仅讨论其中几个方面。关于计算机视觉更多的知识，可以查看相关的书籍。

在本书中，我们将使用python和**opencv**进行计算机视觉相关的工作。**opencv**的全称是Open-Source Computer Vision。它包含了大量可以使用的函数库。OpenCV库本身提供了完整详细的文档，更多内容可以查阅相关的[文档](http://opencv.org/)。

## Canny边缘检测
Canny边缘检测算法是John F.Canny于1986年开发出来的一个边缘检测算法。边缘检测的目的是只是图像中物体的边界。

Canny边缘检测的基本原理就是通过计算图像中各个像素点的梯度，并将梯度值作为像素值，从而有助于发现图像的边缘，以及进一步更容易根据形状检测对象。

接下来，我们首先看一下openCV中Canny边缘检测函数的输入参数：

```
edges = cv2.Canny(gray, low_threshold, high_threshold)
```
在这个函数中，我们将Canny算法用于gray的灰度图像中，并将处理后的图像输出给一个edges变量。上、下阈值决定了边缘被检测得到所需要满足的强度。其中，强度的定义时图像中相邻像素之间的差异程度（这就是所说的梯度）。

接下来，我们通过一个例子来演示一下Canny边缘检测的功能。
下图是一个灰度图像，其中图像中有明亮的部分、暗淡的部分以及亮度快速变化的部分。
![灰度图像](/assets/8.jpg)

整个图像可以看作是一个在二维坐标系下的关于x和y的数学函数：
![二元函数](/assets/9.jpg)
因此，我们可以对这个图像进行一些数学运算。

例如，我们可以对每个像素点的值进行求导：

$$
\frac{df}{dx} = \Delta(pixel\_value)
$$

当导数值越大时，说明图像中在该像素点的颜色变换越明显，反之则说明变化越不明显。由于图像是二维的，因此，我们在计算导数时可以同时计算关于x和y的导数。对于计算导数后得到的图像，我们称之为**梯度图像**。

通过基本的梯度计算，我们可以得到较厚的边缘线。而Canny算法可以帮助我们把边缘线变得更细。此外，我们还可以在调用Canny算法时，选择更小的下限阈值，有助于连接这些较强的边缘。

Ps：在图像处理中，图像的左上角是坐标原点(0,0)。y值向下增加，x值向右增加。



