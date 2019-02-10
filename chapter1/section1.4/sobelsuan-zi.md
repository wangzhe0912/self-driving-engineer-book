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
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
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
    
# Run the function
grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```











