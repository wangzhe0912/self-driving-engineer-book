## HLS

对于其他颜色的车道线（如黄色），或者在不同光照条件下，这种方法立刻不再适用了。

例如，我们仍然以一副交通图像为例：

![黄色车道线](/assets/62.jpg)

左上角的图像为原始图像，可以看到图像中存在一条黄色的车道线。接下来，我们将其拆分为R,G,B三个通道，可以看到，在R，G通道中黄色车道线所在的像素仍然保存了较高的亮度，燃石在B通道的图像中，黄色车道线所在的像素点的亮度甚至低于其他像素点。因此，我们不能直接使用R,G,B三个通道单独进行过滤。

那么，我们来想一下有没有更好的方法来识别车道线。事实上，对于一副图像而言，除了用RGB方式对其表示外，还有其他很多的表示方式。这些不同的颜色表示方式就成为**颜色空间**。

RGB是最常见的一种颜色表示方式：

![RGB空间](/assets/63.jpg)

在这个空间中，任何颜色都可以通过R,G,B的三维坐标值来表示，例如，对于白色而言，可以表示为(255,255,255)

除了RGB空间外，图像分析领域中常见的颜色空间还有HSV颜色空间、HLS颜色空间等。

![](/assets/64.jpg)

HSV颜色空间中，分别表示色相、饱和度和颜色值。
HLS颜色空间中，分别表示色相、亮度和饱和度。

在这两个颜色空间中，其均呈现圆柱形。其中，色相H的取值范围为[0,179]。

以HLS颜色空间为例：L（亮度）主要受到光照条件所影响。而H和S通道则对亮度不敏感。因此，我们如果仅使用H，S两个通道，而丢失L通道信息，就可以就是在不同光照条件下识别车道线不理想的情况了。

下图是将同一张道路图像转换至HLS颜色空间后的结果：

![HLS](/assets/65.jpg)

此时，可以看到在S通道中，我们很清晰的检测到的我们期望寻找的车道线。

Ps：需要说明的是任意一个颜色，都可以用不同的颜色空间进行表示。且不同颜色空间的表示可以通过相应公式进行转换。

### 各个颜色通道对比

首先，我们以一副图像为例来分析灰度、RGB以及HLS在车道线检测上的表现。原始图像如下所示：

![](/assets/66.jpg)

接下来，我们首先根据灰度变化的结果进行过滤：

![](/assets/67.jpg)

接下来，我们从R通道进行处理：

![](/assets/68.jpg)

最后是S和H通道：

![](/assets/70.jpg)

从上述的四个通道的对比可以看出，S通道对于车道检测有着相对更好的效果。事实上，对于白色车道线而言，R通道同样可以获得很理想的效果，在实际应用中，我们可以对各个颜色通道进行结合使用，从而得到更稳定的检测效果。

### HLS实战

接下来，我们将在HLS颜色空间中，使用S通道结合阈值进行车道线检测。

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# 读取图像
image = mpimg.imread('test6.jpg') 

# 使用S通道进行阈值处理
def hls_select(img, thresh=(0, 255)):
    # img：输入图像
    # thresh：阈值
    # 处理步骤：
    # 1) 转换至HLS空间
    # 2) 对S通道进行阈值处理
    # 3) 返回处理后的结果
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    binary_output = np.zeros_like(s)
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary_output

# 调用函数
hls_binary = hls_select(image, thresh=(100, 255))

# 显示
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```

可以看到，利用S通道进行阈值处理后，得到的结果如下：

![](/assets/71.jpg)
