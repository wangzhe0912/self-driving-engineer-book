## 车道线定位

在之前的步骤中，我们已经通过透视变换找到了车道线的鸟瞰图。接下来，我们需要做的是从图像中找出车道线的位置（即找出哪些像素点属于车道线，哪些属于左侧车道线，哪些属于右侧车道线）。有许多找出车道线的方法，下面我们将列出其中的几种方法。

下面，我们以下图为例进行说明：

![](/assets/78.jpg)


### 直方图峰值法

下面，我们首先读取该图像，并对图像的下半部分进行直方图统计：

```python
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# 读取图像并归一化
img = mpimg.imread('warped_example.jpg')/255

def hist(img):
    # 对图像的下半部分进行截取
    bottom_half = img[img.shape[0]//2:,:]
    # 统计每个x坐标下1出现的次数
    histogram = np.sum(bottom_half, axis=0)
    return histogram

# 调用函数
histogram = hist(img)

# 图像可视化
plt.plot(histogram)
```

得到的结果如下：

![histogram](/assets/79.jpg)

从上图中，我们可以看到其存在两个峰值，它们可以用作搜索的起点。

![](/assets/80.jpg)
![](/assets/81.jpg)

接下来，我们可以从起点出发，使用一个滑动窗口，从搜索起点开始，一直找到完整的车道线。

![](/assets/82.jpg)
![](/assets/83.jpg)


### 滑窗法1

接下来，我们可以使用柱状图中的两个最高峰值作为确定车道线位置的起点，然后使用图像中向上移动的滑动窗口（沿道路进一步移动）来确定车道线的位置。

第一步：从直方图中分别找出左右两个车道线的起点。
第二步：设置滑动窗口相关的参数。例如滑动窗口的数量、滑动窗口的宽度、滑动窗口最小像素点等。
第三步：迭代找出全部的窗口。
目前，我们已经可以根据直方图信息找出起始窗口，下面，我们将会通过循环逐层移动窗口来找出完整的车道线。具体流程：

1. 根据滑动窗口的数量进行循环。
2. 找出当前窗口中的边界值。（Ps：可以使用`cv2.rectangle`函数在可视化图像中画出该边界线）
3. 找出窗口内部所有非零的像素点的数量。
4. 记录当前窗口的位置
5. 若3中找到的像素点数量大于我们设置的滑动窗口最小像素点，此时修改当前的窗口位置。

第四步：多项式拟合，即将每一行中找到的坐标点拟合成为一条曲线。

完整实现参考如下：

```python
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

# 加载图像
binary_warped = mpimg.imread('warped_example.jpg')

def find_lane_pixels(binary_warped):
    # binary_warped：经过透视变换后的二值图像
    # 计算图像下半部分的直方图
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # 生成一个输出图像用于可视化显示
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # 根据直方图信息找出左右两个车道线的划窗起点
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 自定义划窗超参数
    nwindows = 9
    margin = 100
    minpix = 50

    # 计算窗的高度
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # 计算图像中所有非零像素点的x，y坐标
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img


out_img = fit_polynomial(binary_warped)

plt.imshow(out_img)
```


### 滑窗法2




### 基于先验信息的查找







