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
    该函数会返回一个单通道的图像
    Ps：需要注意的是，如果想要显示该图像，需要在imshow函数中添加cmap='gray'参数，例如
    plt.imshow(gray, cmap='gray')
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 如果是使用cv2.imread()读取图像，则需要使用BGR2GRAY进行转换，如下：
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
def canny(img, low_threshold, high_threshold):
    """
    使用canny边缘检测算法
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """
    高斯平滑
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    感兴趣区域提取
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # 初始化一个空的感兴趣预期
    mask = np.zeros_like(img)   
    # 根据图像的类型，定义对应的感兴趣区域的填充值
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # 调用fillPoly函数对感兴趣区域填充值
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # 仅保留感兴趣区域的值
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    用于将检测到的车道线显示到图像中的函数
    我们可以通过线段的斜率(y1 - y2) / (x1 - x2)找出哪些线段属于左侧车道线，哪些线段属于右侧车道线。
    接下来，我们可以分别对属于两条车道线的线段位置进行求平均值，然后再扩展至车道的首位两端。
    
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
                # 脏数据
                pass
    # 分别计算左右车道的k和b的平均值，然后将线段在图像中显示出来
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
    找出霍夫变换线
    其中，img应该是Canny边缘检测的输出
    返回结果是经过霍夫变换找到的线段
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    img是通过霍夫变换找到的车道线
    initial_img是原始的图像文件
    处理方式如下：initial_img * α + img * β + γ
    得到的结果是将通过霍夫变换找到的车道线标注到原始图像中。
    Ps：initial_img and img的大小应该一致。
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
```

第四步：列出待测试图像
```python
import os
os.listdir("test_images/")
```
可以看到test_images目录下有如下7副图像。
![7 img](/assets/29.jpg)

第五步：利用之前的辅助函数构建车道线的处理流程
```python
import time
ori_image = mpimg.imread('test_images/solidWhiteRight.jpg')
# Step1: 灰度变化
image = grayscale(ori_image)
plt.imshow(image, cmap='gray')
# Step2: canny边缘检测
time.sleep(5)
image = canny(image, 50, 150)
plt.imshow(image, cmap='gray')

# Step3: 高斯平滑滤波
time.sleep(5)
image = gaussian_blur(image, 9)
plt.imshow(image, cmap='gray')

# Step4: 感兴趣区域提取
time.sleep(5)
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(500, 300), (501, 300), (imshape[1],imshape[0])]], dtype=np.int32)
image = region_of_interest(image, vertices)
plt.imshow(image, cmap='gray')

# Step5: 霍夫变换
time.sleep(5)
rho = 2  # distance resolution in pixels of the Hough grid
theta = 2 * np.pi/180  # angular resolution in radians of the Hough grid
threshold = 80     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 40  #minimum number of pixels making up a line
max_line_gap = 3    # maximum gap in pixels between connectable line segments
image = hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)
plt.imshow(image, cmap='gray')

# Step6：最终结果展示
time.sleep(5)
image = weighted_img(image, ori_image, α=0.8, β=1., γ=0.)
plt.imshow(image)
```
可以看到，经过一系列处理后，得到的结果如下：
![processed_img](/assets/30.jpg)

第六步：基于视频的处理
下面，我们将会完成一个更Cool的事情，在视频上标注车道线~代码库中提供了两个视频可以用于测试你的算法。
Ps：对于视频的处理需要依赖于ffmpeg exe文件，可以通过如下方式进行下载。
```python
imageio.plugins.ffmpeg.download()
```
下面，我们先将之前对图像的处理封装为一个函数：
```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
def process_image(image):
    # Step1: gray_image
    gray_image = grayscale(image)
    
    # Step2: canny
    canny_image = canny(gray_image, 50, 150)

    # Step3: gaussian blur
    gaussian_blur_image = gaussian_blur(canny_image, 5)

    # Step4: region_of_interest
    imshape = gaussian_blur_image.shape
    vertices = np.array([[(0,imshape[0]),(500, 300), (501, 300), (imshape[1],imshape[0])]], dtype=np.int32)
    interest_region_image = region_of_interest(gaussian_blur_image, vertices)

    # Step5: hough_lines
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = 2 * np.pi/180  # angular resolution in radians of the Hough grid
    threshold = 80     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40  #minimum number of pixels making up a line
    max_line_gap = 3    # maximum gap in pixels between connectable line segments
    # line_image = np.copy(image)*0 
    lanes_image = hough_lines(interest_region_image, rho, theta, threshold, min_line_len, max_line_gap)
    
    # Step6: weighted_img
    result = weighted_img(lanes_image, image, α=0.8, β=1., γ=0.)
    return result
```
下面，我们将该函数应用到视频中：
```python
white_output = "test_videos_output/solidWhiteRight.mp4"
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
# 为了加快测试过程，您可能希望在较短的视频中进行测试和验证
# 此时，我们可以在VideoFileClip函数返回对象截取，例如我们可以截取前5s的视频
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
white_clip = clip1.fl_image(process_image)
%time white_clip.write_videofile(white_output, audio=False)
# 显示该视频
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```

第七步：一个更艰巨的挑战
接下来，你可以将你的处理函数用于`test_videos/challenge.mp4`视频，看看效果如何。
![pool](/assets/31.jpg)
很明显的是，我们目前的算法对于弯道的车道线标记并不理想，有没有什么办法解决呢？继续接下来的学习吧~


