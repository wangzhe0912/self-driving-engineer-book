## 透视变换

### 车道线的曲率
现在，我们已经学习了如何进行图像校准，下面，我们将要学习的是如何去感知和计算车道线的曲率。车道线的曲率计算对于无人驾驶汽车而言至关重要，无人驾驶汽车需要根据车道线的曲率来判断转向角的大小，从而才能进行左拐或右拐。

实际上，我们通过汽车行驶的速度以及车道的弯曲程度，就可以轻松的计算出转向角的大小。因此，我们将在本节中学习如何计算车道线的曲率。

计算车道线的曲率的步骤简述如下：

Step1：使用感兴趣区域提取和阈值化等技术来检测出车道线。
![Step1](/assets/43.jpg)

Step2：通过透视变换获得车道的俯视图。
![Step2](/assets/44.jpg)

Step3：通过多项式拟合来找出车道线的表示方式
Step4：根据拟合的多项式获得车道线的曲率
![Step4](/assets/45.jpg)

### 透视变换原理

透视变换是图像成像中一个现象。
例如：
1. 物理距离观察点越远，则看起来越小。反之，物体距离观察点越近，则看起来越大。
2. 平行的直线在远处似乎会汇聚到同一个点上。

换做数学的表达方式来讲，当我们将真实世界看作三维坐标，z轴可以表示物体距离观察点的远近。当z值越大，即物理距离我们越远，看起来就越小。

透视变换就是通过改变z轴的坐标，从而改变了对象的二维图像表示，通过改变观察视角来改变图像的二维表示。

如下图所示，变换前：
![变换前](/assets/46.jpg)
变换后：
![变换后](/assets/47.jpg)

例如，我们希望对公路图像进行透视变换时，我们可以放大图像中距离较远的物体，如下图所示：
![透视变换](/assets/48.jpg)
透视变换一个非常有用的工具，在某些任务中（例如车道线曲率计算）通过上帝视角（鸟瞰图）更加容易进行。

透视变换的处理过程与图像校准类似，唯一的区别在于：图像校准是将扭曲点映射到真实物体对应点；而透视变换则是将给定图像点映射到新视角下的图像点。

![映射关系](/assets/49.jpg)

**Question**：对于从汽车前置摄像头拍摄的道路图像，我们为什么希望进行透视变换？
A. 鸟瞰图更容易计算梯度
B. **鸟瞰图有助于我们衡量车道线的曲率**
C. 透视变换有助于消除我们测量过程中引入的弯曲误差
D. 透视变换有助于消除道路颜色、阴影差异


### 透视变换实战
下图是一个通过某视角拍摄到的交通信号牌：
![交通信号牌](/assets/50.jpg)
接下来，我们将会演示如果对该信号牌图像进行透视变换，从而使该图像看起来像是在正前方拍摄得到的：
![正前方拍摄](/assets/51.jpg)

为了进行透视变换，我们需要再原始图像中选择四个点，这些点在原始图像中属于同一平面且组成了一个长方形。
Ps：四个点就足够帮助我们从一种视角转化为另外一种视角了。
此外，我们还需要定义期望所选的四个点在变化后所出现的位置。
然后，我们就可以使用OpenCV函数来计算这种透视变换了。
如下图所示：
![透视变换](/assets/52.jpg)

完整实现如下：
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('stopsign.jpg')

img_size = (img.shape[1], img.shape[0])

# 原始图像中的四个点
src = np.float32([
  [850, 320],
  [865, 450],
  [533, 350],
  [535, 210]
])

# 映射至新视角后对应的四个点
dst = np.float32([
  [870, 240],
  [870, 370],
  [520, 370],
  [520, 240]
])

# 计算透视变换矩阵M
M = cv2.getPerspectiveTransform(src, dst)

# 对原始图像使用变换矩阵
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR) # INTER_LINEAR线性内插法补点

# 显示变换后的图像
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.set_title("source")
ax1.imshow(img)
ax2.set_title("warpped")
ax1.imshow(warped)
```

Ps: 应用透视变换时，手动选择四个源点通常不是最佳选择。实际上，还有许多其他方法可以帮助我们选择源点。例如，许多透视变换算法将基于边缘或角点检测以及分析诸如颜色和周围像素的属性以编程方式检测图像中的四个源点。

**Question**: 摄像机校准，失真校准和透视变换的作用分别是什么？

1. 摄像机校准：用于计算从3维物体到2维平面图像的转换公式。
2. 失真校准：确保观测物体的几何形状不会因为所在图像中不同的位置而产生变化。
3. 透视变换：能够通过不同的视角和方向来观察物体。


### 结合图像校准与透视变换
接下来，我们将结合这两节的内容，对一副图像先进行图像校准后再进行透视变换，最终得到结果图像，如下图所示：
![](/assets/53.jpg)

具体的步骤如下：

1. 使用`cv2.undistort()`函数对图像进行校准
2. 将图像转换为灰度图像
3. 找出棋盘的内角
4. 在图像中标记出棋牌的内角
5. 定义原始图像中的四个点作为标记点
6. 设置目标转换图像中四个点的位置（需要与原始图像一一对应）
7. 使用`cv2.getPerspectiveTransform()`函数计算转换矩阵M。
8. 使用`cv2.warpPerspective()`函数对原始图像进行变换。

完整实现如下：
```python
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # 图像校准与透视变换
    offset = 100
    copy_img = np.copy(img) 
    undist = cv2.undistort(copy_img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        plt.imshow(img)
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        img_size = (undist.shape[1], undist.shape[0])
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```
