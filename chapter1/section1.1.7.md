## Canny边缘检测在车道线检测中的应用

了解了Canny边缘检测的基本原理后，接下来，我们将通过编程来将Canny边缘检测应用于车道线检测的应用中。

第一步，我们需要读取一副图像：
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread('exit-ramp.jpg')
plt.imshow(image)
```
![CannyImage1](/assets/11.jpg)

我们加载了一副道路的图像，人眼观察的话我们很容易找到车道线在哪里，但是如何让计算机程序做到这一点呢？

第二步，我们需要将图像转换为灰度图像：
```python
import cv2  #bringing in OpenCV libraries
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion
plt.imshow(gray, cmap='gray')
```
![CannyImage2](/assets/12.jpg)

第三步，Canny边缘检测
接下来，我们将会将Canny边缘检测算法应用于这副图像中。我们来看一下OpenCV中Canny边缘检测函数：
```python
edges = cv2.Canny(gray, low_threshold, high_threshold)
```
通过该函数，我们将对输入的灰度图像进行canny边缘检测处理，输出的是应用边缘检测算法后得到的图像`edges`。

在canny边缘检测算法中，首先会找出梯度高于`high_threshold`的像素点作为高概率边缘，同时过滤掉梯度低于`low_threshold`的像素点。接下里，对于梯度值在`low_threshold`和`high_threshold`之间的像素点，如果该像素点与高概率边缘相邻，则保留该像素点，否则则过滤该像素点。

对于输出的图像`edges`，其中检测到的边缘将会显示为白色，其余像素点均为黑素。

那么，`low_threshold`和`high_threshold`应该如何选择呢？由于我们将图像转为了灰度图像，每个像素点的值由8bits组成。因此，`low_threshold`和`high_threshold`的合法输入为[0, 255]。

而`threshold`的实际含义表示相邻像素点之间的颜色差异，因此，一个合理的选择应该在几十到一两百之间。根据经验来看，`low_threshold`和`high_threshold`之前的大小比例应该在1:2或1:3。













