## 项目解决方案

怎么样？是否成功的完成了任务呢？下面，我们来给出一个解决方案，对比一下，看看哪个效果更好呢！

在上面的内容中，我们已经学习了颜色选择、感兴趣区域选择、灰度变换、高斯平滑、Canny边缘检测和霍夫变换等技术。

我们将会结合直接所学习的各项技术来完成对视频中车道线的检测。
Ps：对视频车道线的检测与对图像车道线的检测非常类似，我们首先针对图像编写处理流程，然后将该流程应用于视频中的每一帧即可。

此外，我们在图像检测中，首先会找出图像中车道线的位置，如下图所示：
![车道线位置](/assets/26.jpg)
接下来，则会将找出的车道线进行连接，得到如下的图像：
![车道线连线](/assets/27.jpg)

下面，我们将在jupyter-notebook中一步步的来实现上述描述的功能。

第一步，引入相关的包：
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```

第二步：读取原始图像
```python
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
```
![read_img](/assets/28.jpg)

可以看到，读取图像的维度为$$540 * 960 * 3$$。

第三步：**helper_function**
接下来，我们将会列举出之前文章中所使用的各个函数，这些函数将有效的帮助我们实现上述功能。
```python

```
