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


