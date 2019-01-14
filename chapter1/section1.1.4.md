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

