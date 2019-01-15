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
