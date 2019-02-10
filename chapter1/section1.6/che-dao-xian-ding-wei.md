## 车道线定位

在之前的步骤中，我们已经通过透视变换找到了车道线的鸟瞰图。接下来，我们需要做的是从图像中找出车道线的位置（即找出哪些像素点属于车道线，哪些属于左侧车道线，哪些属于右侧车道线）。有许多找出车道线的方法，下面我们将列出其中的几种方法。

下面，我们以下图为例进行说明：

![](/assets/78.jpg)


### 直方图峰值法

下面，我们首先读取该图像，并对图像的下半部分进行直方图统计：

```python
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# 读取图像并归一化
img = mpimg.imread('warped_example.jpg')/255

def hist(img):
    # 对图像的下半部分进行截取
    bottom_half = img[img.shape[0]//2:,:]
    # 统计每个x坐标下1出现的次数
    histogram = np.sum(bottom_half, axis=0)
    return histogram

# 调用函数
histogram = hist(img)

# 图像可视化
plt.plot(histogram)
```

得到的结果如下：

![histogram](/assets/79.jpg)





### 滑窗法1



### 滑窗法2




### 基于先验信息的查找







