# 1.1 计算机基础

为了让车辆能够自主的行驶，我们首先需要让车辆能够正常感知到车辆周围的环境信息。
在我们人类驾驶的过程中，我们是通过眼睛来观察车道线的位置，从而判断在哪里转弯，如何正常行驶等。而在自动驾驶的汽车中，我们需要使用摄像头和一些其他类型的传感器来实现类似的功能。我们将要解决的第一个问题就是如何去识别车道线。


下图是一个车载摄像头拍摄到的图片，观察这个图片，你觉得哪些特征可以帮助我们来定位车道线呢？
![前置摄像头](/assets/2.jpg)

事实上，我们可以从以下四个方面来帮助我们定位车道线：

1. 颜色
2. 形状
3. 方向
4. 位置


## RGB基本概念
首先，颜色是有助于我们识别车道线的最显著特征。通常，车道线的颜色是白色或黄色的。
以白色为例，我们应该如何在图像中找出白色的车道线呢？
对于一副RGB图像而言，分别由三个颜色通道的数据组成。R表示Red，G表示Green，B表示Blue。而一切颜色都构成都是通过调整三个通道各自的值而得到的。
![RGB图像](/assets/3.jpg)
其中，每个颜色通道的像素值的范围都是在0到255之间。其中，0表示最暗，255则表示最亮。

那么，思考一下图像中白色的车道线应该怎么用三个颜色通道[R,G,B]的数字来表示最为合适呢？

A. [0,0,0]
B. [0,255,255]
C. [100,150,200]
D. [255,255,255]

Yeah！你答对啦~是的，[255,255,255]表示白色~

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

## 感兴趣区域提取

现在，我们不再仅仅关注于图像的颜色，而是考虑图像中我们感兴趣的区域，即车道线所在的大致区域范围。在本例中，我们的图像是由车辆的前置摄像头拍摄得到的，因此，车道线所在的位置大致如橘色区域所示：
![感兴趣区域](/assets/6.jpg)
因此，我们可以仅仅关注橘色线段所包围的区域。在图中，我们用一个三角形表示了我们感兴趣的区域（当然，你也可以自由定义你认为的感兴趣区域的形状和位置），下面，我们将会用Python代码来实现仅仅保留感兴趣区域的内容，而过滤到所有感兴趣区域外的信息。

同样，我们需要先引入相关的库并读取图片信息：
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('attachment/rgb_image.jpg')
print('This image is: ', type(image), 'with dimensions:', image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
# Note: always make a copy rather than simply using "="
region_select = np.copy(image)
```

接下来，为了定义第一个三角形的感兴趣区域，我们需要在图像中选择三个点作为三角形的三个顶点：
```python
left_bottom = [0, 430]
right_bottom = [760, 430]
apex = [380, 240]
```

下面，我们需要通过三角形的三个顶点来计算得到三角形的三条边函数：
```python
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
```

下面，我们需要依次判断图像中的每个像素点是否属于三角形的三条边的内部，其中属于三角形区域内部转换为数学表达式后即为在三角形左、右侧边的下方且在三角形下侧边的上方的交集（Ps：其中左上角是原点）：
```python
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY < (XX*fit_left[0] + fit_left[1])) | (YY < (XX*fit_right[0] + fit_right[1])) | (YY > (XX*fit_bottom[0] + fit_bottom[1]))
```

最后，我们可以将所有非感兴趣区域设置为黑色：
```python
region_select[region_thresholds] = [0, 0, 0]

# Display the image
plt.imshow(region_select)

plt.show()
```
处理完成后，得到的图像如下：
![感兴趣区域提取图像](/assets/7.jpeg)


## 结合感兴趣区域提取与颜色过滤器提取车道线
上面的过程中，我们已经学习了如何分别使用颜色过滤器和感兴趣区域提取对图像处理，下面，我们来看下如何结合两种方式来更加精确的检测出车道线：
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('attachment/rgb_image.jpg')

ysize = image.shape[0]
xsize = image.shape[1]
color_select= np.copy(image)
line_image = np.copy(image)

# Define our color criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

left_bottom = [0, 430]
right_bottom = [760, 430]
apex = [380, 240]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | (image[:,:,1] < rgb_threshold[1]) | (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & (YY > (XX*fit_right[0] + fit_right[1])) & (YY < (XX*fit_bottom[0] + fit_bottom[1]))
# Mask color selection
color_select[color_thresholds] = [0, 0, 0]
# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# Display our two output images
plt.imshow(color_select)
plt.imshow(line_image)

# uncomment if plot does not display
plt.show()
```
通过上述代码，我们可以得到处理后的图像如下：
![结合感兴趣区域提取与颜色过滤器提取车道线](/assets/8.jpeg)

目前为止，我们已经可以找到简单图像的车道线了，但实际上，目前的算法距离真实的应用还有很大的差距。事实上，车道线并不总是白色的，还有黄色的；此外，在不同的条件下（白天、夜晚、雨天等等）即使是相同颜色的车道线也无法通过我们简单的颜色过滤算法检测到。

接下来，我们需要通过更加复杂的计算机视觉方法来检测任何有颜色的线条。

## 计算机视觉
既然我们已经提到了计算机视觉，那么计算机视觉究竟是什么呢？
计算机视觉实际就是通过利用各种算法，使得计算机能够向我们人类一样看到一个有颜色、形状的世界。计算机视觉是一个很大的领域，在本书中，我们仅讨论其中几个方面。关于计算机视觉更多的知识，可以查看相关的书籍。

在本书中，我们将使用python和**opencv**进行计算机视觉相关的工作。**opencv**的全称是Open-Source Computer Vision。它包含了大量可以使用的函数库。OpenCV库本身提供了完整详细的文档，更多内容可以查阅相关的[文档](http://opencv.org/)。

## Canny边缘检测
Canny边缘检测算法是John F.Canny于1986年开发出来的一个边缘检测算法。边缘检测的目的是只是图像中物体的边界。

Canny边缘检测的基本原理就是通过计算图像中各个像素点的梯度，并将梯度值作为像素值，从而有助于发现图像的边缘，以及进一步更容易根据形状检测对象。





















