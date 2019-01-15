## Canny边缘检测在车道线检测中的应用

了解了Canny边缘检测的基本原理后，接下来，我们将通过编程来将Canny边缘检测应用于车道线检测的应用中。

第一步，我们需要读取一副图像：
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread('exit-ramp.jpg')
plt.imshow(image)
```
![CannyImage](/assets/11.jpg)



