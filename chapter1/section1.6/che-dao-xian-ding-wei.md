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

从上图中，我们可以看到其存在两个峰值，它们可以用作搜索的起点。

![](/assets/80.jpg)
![](/assets/81.jpg)

接下来，我们可以从起点出发，使用一个滑动窗口，从搜索起点开始，一直找到完整的车道线。

![](/assets/82.jpg)
![](/assets/83.jpg)


### 滑窗法1

接下来，我们可以使用柱状图中的两个最高峰值作为确定车道线位置的起点，然后使用图像中向上移动的滑动窗口（沿道路进一步移动）来确定车道线的位置。

第一步：从直方图中分别找出左右两个车道线的起点。
第二步：设置滑动窗口相关的参数。例如滑动窗口的数量、滑动窗口的宽度、滑动窗口最小像素点等。



### 滑窗法2




### 基于先验信息的查找







