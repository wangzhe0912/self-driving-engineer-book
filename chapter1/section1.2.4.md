## 项目解决方案

怎么样？是否成功的完成了任务呢？下面，我们来给出一个解决方案，对比一下，看看哪个效果更好呢！

在上面的内容中，我们已经学习了颜色选择、感兴趣区域选择、灰度变换、高斯平滑、Canny边缘检测和霍夫变换等技术。

我们将会结合直接所学习的各项技术来完成对视频中车道线的检测。
Ps：对视频车道线的检测与对图像车道线的检测非常类似，我们首先针对图像编写处理流程，然后将该流程应用于视频中的每一帧即可。

此外，我们在图像检测中，首先会找出图像中车道线的位置，如下图所示：
![车道线位置](/assets/26.jpg)
接下来，则会将找出的车道线进行连接，得到如下的图像：
![车道线连线](/assets/27.jpg)

下面，我们将在jupyter-notebook中一步步的来实现上述描述的功能。（基于python3）

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
import math

def grayscale(img):
    """
    灰度变化函数
    Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_list = []
    right_list = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y1 - y2) / (x1 - x2)
            if 0.5 < k < 0.7:
                # right
                right_list.append([x1, y1, x2, y2])
            elif -0.7 < k < -0.5:
                # left
                left_list.append([x1, y1, x2, y2])
            else:
                # stash data
                pass
    # separate calculate the sum of k, b of left line and right line
    try:
        left_k_sum = 0
        left_b_sum = 0
        left_min_point = (1000, 0)
        left_max_point = (0, 0)
        for line in left_list:
            x1, y1, x2, y2 = line
            k = (y1 - y2) / (x1 - x2)
            b = y2 - k * x2
            left_k_sum += k
            left_b_sum += b
            if x1 < left_min_point[0]:
                left_min_point = (x1, y1)
            if x2 > left_max_point[0]:
                left_max_point = (x2, y2)
        left_k_average = left_k_sum / len(left_list)
        left_b_average = left_b_sum / len(left_list)
        left_min_x = int((img.shape[0] - left_b_average) / left_k_average)
        cv2.line(img, (left_min_x, img.shape[0]), (left_max_point[0], left_max_point[1]), color, thickness)
    except Exception as e:
        print(str(e))

    try:
        # separate record the point of left line and right line
        right_k_sum = 0
        right_b_sum = 0
        right_min_point = (1000, 0)
        right_max_point = (0, 0)
        for line in right_list:
            x1, y1, x2, y2 = line
            k = (y1 - y2) / (x1 - x2)
            b = y2 - k * x2
            right_k_sum += k
            right_b_sum += b
            if x1 < right_min_point[0]:
                right_min_point = (x1, y1)
            if x2 > right_max_point[0]:
                right_max_point = (x2, y2)
        right_k_average = right_k_sum / len(right_list)
        right_b_average = right_b_sum / len(right_list)
        # left_min_point and right_max_point can be calculate by k, b and image shape
        right_max_x = int((img.shape[0] - right_b_average) / right_k_average)
        cv2.line(img, (right_min_point[0], right_min_point[1]), (right_max_x, img.shape[0]), color, thickness)
    # print("Left line: " + str((left_min_x, img.shape[0])) + " " + str(left_max_point) + " " + str(left_k_average) + " " + str(left_b_average))
    # print("Right line: " + str(right_min_point) + " " + str((right_max_x, img.shape[0])) + " " + str(right_k_average) + " " + str(right_b_average))
    except Exception as e:
        print(str(e))

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
```
