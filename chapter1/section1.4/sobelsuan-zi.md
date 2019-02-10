## Sobel算子

Sobel算子是在之前章节中使用的canny边缘检测算法的核心。将Sobel运算符应用于图像是获取图像在X或Y方向上的导数的一种常见方法。其中，Sobel_x和Sobel_y的表达式如下：
![Sobel](/assets/55.jpg)

上述表达式是核大小为3的Sobel算子的示例。核的大小最小是3，但实际上，核大小可以是任何奇数。较大的内核意味着在图像的较大区域上计算梯度，也就是使用更平滑的梯度。

为了理解如何通过这些算子计算导致，我们可以考虑将其中一个算子叠加到图像的3x3区域上。如果图像在该区域是平滑的（即给定区域的值变化很小），则计算结果（将算子的元素与图像元素对应相乘并求和）将为零。计算公式如下：
$$
gradient = \sum(region * S_x)
$$

当我们对于图像中的某个点应用Sobel_x算子时，得到的结果越小，表明该像素点在x方向的梯度越小；同理。对于图像中的某个点应用Sobel_y算子时，得到的结果越小，表明该像素点在y方向的梯度越小。

下面，我们来看一个示例，原始图像如下：
![](/assets/56.jpg)
经过Sobel_x和Sobel_y算子处理后，得到的图像分别如下：
![](/assets/57.jpg)

在上面的图像中，我们可以看到在X和Y方向上进行的梯度检测车道线并得到的边缘结果。采用Sobel_x得到的边缘线趋近于垂直；采用Sobel_y得到的边缘线趋近于水平。

### Sobel算子实战

在接下来的示例中，我们将用Python代码实现利用Sobel算子进行图像滤波，其中既包括Sobel_x算子，也包括Sobel_y算子。

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# 读取图像
image = mpimg.imread('signs_vehicles_xygrad.png')

# 定义一个sobel算子的阈值函数
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # img：输入图像
    # orient：Sobel算子方向
    # thresh_min：滤波最小阈值
    # thresh_max：滤波最大阈值
    # 具体步骤
    # 1) 灰度变换
    # 2) 根据输入方向求导数
    # 3) 对导数的值取绝对值
    # 4) 将像素点的值归一化至0-255
    # 5) 将阈值范围内的像素点设置为1，其他像素点设置为0
    # 6) 返回结果
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

# 调用定义好的函数
grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
# 绘制图像
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```

通过上述代码，可以得到的结果如下：


### 梯度的方向

在上面的实战中，我们通过对梯度进行阈值化，可以较好的找到车道线；但是在这一过程中，还是有一些其他物体没有被过滤掉。

实际上，根据梯度值进行阈值化过滤是Canny边缘检测算法的核心，这也是Canny边缘检测的核心手段。

然而，对于车道线检测问题而言，我们更加关心的是某个特定方向的边界线，因此，我们可以更近一步的考虑梯度的方向。

梯度的方向计算公式如下：

$$
arctan(sobel_y / sobel_x)
$$

上式表示梯度的方向为y方向的地图值 / x方向的梯度值的反正切。

从而，对元素图像中每一个像素点进行上述公式计算，我们可以得到该像素点的梯度方向，其范围为$$[-\pi/2, \pi/2]$$。其中，0表示垂直线，$$-\pi/2, \pi/2$$则表示水平线。

接下来，我们将会编写一个函数用于计算像素点的梯度，然后用于一个阈值对其进行过滤，从而观察一下梯度方向的特性。

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 读取图像
image = mpimg.imread('signs_vehicles_xygrad.png')

# 定义一个计算梯度方向并过滤的函数.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # img：原始图像
    # sobel_kernel：核大小
    # thresh：阈值
    # 处理步骤
    # 1) 灰度变换
    # 2) 分别计算x方向和y方向的梯度
    # 3) 对两个方向的梯度取绝对值
    # 4) 计算梯度方向
    # 5) 根据阈值进行过滤
    # 6) 返回过滤后的图像
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    gradient_sobel = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(gradient_sobel)
    binary_output[(gradient_sobel >= thresh[0]) & (gradient_sobel <= thresh[1])] = 1
    return binary_output
    
# 调用该函数
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
# 绘图
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```

得到的结果如下：

![梯度方向](/assets/58.jpg)







