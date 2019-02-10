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
3. 找出窗口内部所有非零的像素点。
4. 记录当前窗口下所有找到的非零像素点的位置。
5. 若3中找到的像素点数量大于我们设置的滑动窗口最小像素点，此时修改新的窗口位置为当前所有找到的非零像素点的中心位置。

第四步：多项式拟合，即将所有找到的位于窗口内部的像素点拟合成为一条曲线。

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
    leftx_current = leftx_base
    rightx_current = rightx_base
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

    # 初始化两个数组用于存储左右两个车道线的索引
    left_lane_inds = []
    right_lane_inds = []

    # 根据滑动窗口的数量进行循环
    for window in range(nwindows):
        # 找出每个窗口的边界点
        win_y_low = binary_warped.shape[0] - (window+1)*window_height # 上边缘
        win_y_high = binary_warped.shape[0] - window*window_height    # 下边缘
        win_xleft_low = leftx_current - margin                        # 左车道线左边缘
        win_xleft_high = leftx_current + margin                       # 左车道线右边缘
        win_xright_low = rightx_current - margin                      # 右车道线左边缘
        win_xright_high = rightx_current + margin                     # 右车道线右边缘
        
        # 在可视化图像中画出当前的窗口
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # 找出当前窗口中非零点的x,y坐标集合
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # 将这些点的索引添加至之前创建的数组中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # 连接数组
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # 忽略异常点
        pass

    # 提前出左、有车道线的像素点位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # 找出车道线的坐标点
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # 对左右两个车道线进行多项式拟合
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 生成一组x，y坐标点用于绘制曲线
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # 异常捕获
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # 车道线可视化
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img


out_img = fit_polynomial(binary_warped)

plt.imshow(out_img)
```


### 滑窗法2

另外一种实现滑窗的方式是借助于卷积计算。卷积计算是将两个单独的信号进行相乘后并求和，在我们这个例子中，则是将窗模板和像素图像进行卷积。

将窗口模板从左到右在图像上滑动，依次进行卷积计算，卷积得到的峰值就是像素的最高重叠点以及车道线最可能出现的位置。

现在，我们用具体的代码来实现通过卷积在一个经过阈值处理的道路图像中找到最佳的窗口中心位置。


### 基于先验信息的查找

现在，您已经构建了一个算法使用滑动窗口找出车道线。但是，该算法需要在每一帧上重新开始计算。这看起来效率很低，因为其实在相邻的帧之前，车道线的位置通常不会发生太大的变换。

实际上，在下一帧图像中，我们不需要再次进行盲搜索，而是可以在上一帧车道线位置周围的空白处进行搜索，如下图中所示。绿色阴影区域显示了我们这次搜索线条的位置。因此，一旦你知道这些线在上一帧视频中的位置，你就可以在下一帧中对它们进行高目标的搜索。

![](/assets/84.jpg)

这相当于为每一帧视频使用一个定制的感兴趣区域，它可以帮助您通过急弯和一些特殊条件下车道线检测任务。但是，如果在某些步骤中无法找到车道线时，可以再次采用滑窗法来进行车道线检测。

下面，我们来通过具体的代码进行实践：





