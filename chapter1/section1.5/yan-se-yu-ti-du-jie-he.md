## 颜色与梯度结合

在本节的最后，我们将结合1.4节所讲的梯度检测方法与本节所讲的颜色阈值过滤方法进行车道线检测。

期望效果如下：

![](/assets/72.jpg)

其中，左上角为原始图像，右上角表示用S颜色通道进行过滤后得到的图像，左下角表示通过梯度方法进行过滤得到的图像，最终右下角的图像表示经过二者结合后得到的结果。

对于右上角的图像而言，仍然可以检测树木或汽车周围的边缘。但是这些并不会影响检测效果，因为这些线主要最终可以通过对图像进行感兴趣区域提取级别，基本上可以裁剪掉车道线之外的区域。最重要的目的是，在不同程度的日光和阴影下，可靠地检测不同颜色的车道线。

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image = mpimg.imread('bridge_shadow.jpg')

# 定义一个用于车道线检测的完整函数
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # img：输入图像
    # s_thresh：S通道的阈值
    # sx_thresh：Sobel梯度的阈值
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    # combined_binary
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

# 调用该函数
result = pipeline(image)

# 绘制图像
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(result)
ax2.set_title('Pipeline Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```
