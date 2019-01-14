## 颜色过滤器
了解了什么是RGB图像，同时也知道了在一副RGB图像中白色的表示。那么，接下来我们将会通过编写一段Python代码来实现简单的颜色过滤。
Ps：本书中用到的所有代码及附件均可以在[github代码库](https://github.com/wangzhe0912/self-driving-engineer-book)中找到。

在解决这个问题时，我们首先需要引入两个Python的第三方库：numpy与matplotlib。其中，numpy是Python中最常用的数值计算库，适用于矩阵处理等。matplotlib则是Python中最流行的图像可视化库。

关于二者的学习可以参考相关文档：[NumPy](http://www.numpy.org/)，[matplotlib](https://matplotlib.org/)。

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
```

接下来，我将会使用相关的库函数来读取原始图像并打印一些图像的基本信息：

```python
image = mpimg.imread('attachment/rgb_image.jpg')
print('This image is: ', type(image), 'with dimensions:', image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
# Note: always make a copy rather than simply using "="
color_select = np.copy(image)
```

Ps：在Python语言中，我们通常需要使用copy函数而不是=直接进行赋值。原因在于copy相当于值传递，修改新的变量不会对原始变量有影响；而直接使用=进行赋值时则相当于引用传递，此时，如果直接修改新的变量，同时也会修改了原始的变量。

接下来，我们需要定义三个变量：red_threshold，green_threshold与blue_threshold，并把三个变量组成一个列表rgb_threshold。这三个阈值分别表示在进行颜色过滤时，我们选择的边界阈值。

```python
red_threshold = 0   # 此处仅仅是假设为0，具体的值需要你自己来调试从而找到合适的选择
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]
```

下面就是颜色过滤器中最重要的步骤了，我们需要将R、G、B通道中所有不满足阈值限制的点全部置黑，仅保留满足阈值限制点的原始颜色。

```python
# Identify pixels below the threshold
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]

# Display the image                 
plt.imshow(color_select)
plt.show()
```
其中，`color_select`表示处理后的图像，其中针对RGB图像均满足阈值要求的点被保留了下来，其余点均被置为了黑色。

在上面的代码中，所有阈值均被我们设置为0，此时，所有的点均满足阈值限制，因此输出图像与原始的输入图像是一致的。接下来，你需要调整选择合适的阈值，从而能够使得图像尽量仅保存下白色的车道线。如下图所示：

![颜色过滤器处理后的图像](/assets/5.jpg)

看到上述通过颜色过滤器处理后的图像，可以看到通过简单的颜色选择，我们已经消除了图像中除车道线以外的大部分内容。然而，至此为止我们仍然很难自动得到精确的车道线，因为目前在外围检测到一些不是车道线的相关内容。此时，我们需要借助一些其他技术来进一步处理。

