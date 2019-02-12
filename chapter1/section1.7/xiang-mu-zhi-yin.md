## 项目指引

在本项目中，我们将要利用本章学习的全部技术来完成车道线的识别和跟踪的任务。

### 摄像头校准

在实战项目中，我们使用的棋盘校准图像是9*6的大小，这与之前的章节中用到的棋盘并不完全一致。

### 曲率半径

在本项目中，我们需要计算车道的曲率半径（单位是米，而不是像素）。
下图是来自Google地图的图片，其中包含项目视频的制作地点（位于Udacity办公室的西北部！）。在这里，我画了一个圆圈，与项目视频中的第一条左曲线重合。这是一个非常粗略的估计，该圆的半径约为1km。这个数字可以用于粗略的估计你在项目中所计算得到的曲率半径是否正确。

![](/assets/89.jpg)

### 车辆偏移量

在本项目中，我们同时期望计算出当前车道与车道线中心的偏离的距离。此时，我们可以假设摄像机安装在汽车的中央，这样车道中心就是您检测到的两条线路之间图像底部的中点。车道中心距图像中心的偏移（从像素转换为米）是距离车道中心的距离。

### 车道线追踪

当我们将处理流程应用于视频流处理后，我们希望借助之前找到的车道线应用后续视频中的车道线定位。此时，推荐可以定义一个Line类用于追踪车道线相关的参数。我们可以针对左右车道线分别创建Line对象，并进行车道线跟踪和正确性验证。

```python
class Line():
    def __init__(self):
        # 在上次迭代中是否正常检测到车道线
        self.detected = False  
        # 最近几次检测中的x值
        self.recent_xfitted = [] 
        # 最近几次检测中的x平均值
        self.bestx = None     
        # 最近几次检测中多项式拟合的平均值
        self.best_fit = None  
        # 最近几次检测中多项式拟合的值
        self.current_fit = [np.array([False])]  
        # 曲率半径
        self.radius_of_curvature = None 
        # 车辆偏移距离
        self.line_base_pos = None
        # 两次拟合系数之间的差异
        self.diffs = np.array([0,0,0], dtype='float') 
        # 车道线的x坐标
        self.allx = None  
        # 车道线的y坐标
        self.ally = None  
```

### 正确性验证

当我们的算法找到了对应的车道线后，我们应该检测一下找到的车道线是否存在明显问题，例如：

1. 两条车道线是否具有相似的曲率。
2. 两条车道线是否基本平行。
3. 两条车道线的水平距离是否符合预期。


### 先验搜索

当我们在某一帧视频中找到了车道线，并且确信该找出车道线符合预期，那么就不需要在下一帧中盲目搜索，而是只需在前一次检测周围的窗口内搜索即可。

### 失败重置

当正确性验证不通过时，我们可以简单的假设当前视频可以有损，此时，我们可以先保留之前视频帧得到的结果。但是，如果存在连续几帧信息丢失的情况时，我们则应该重新使用直方图、滑动窗口等方法检测车道线。

### 平滑处理

为了让我们找到的车道线能够逐帧平滑变换，一种推荐的方式是将最后n帧视频中找到的车道线进行平均处理。每次获得新的高置信度测量值时，您都可以将其附加到最近测量值列表中，然后在过去的测量值中取平均值，以获得要在图像上绘制的车道位置。

### 结果可视化

一旦我们在投影空间中找到车道线后，就可以将测量结果投射到路面上图像中了！假设投影空间的二值图像叫做`warped`，我们用线条拟合多项式得到了名为ploty，left_fitx和right_fitx的数组，它们代表线条的x和y像素值。然后，您可以将这些线投影到原始图像上，如下所示：

![](/assets/90.jpg)

实现代码如下：

```python
# 创建用于绘图的图像
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# 将x,y点转换为可用格式
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# 在空白图像中绘制两条车道线
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# 使用逆转换矩阵Minv将其转换为原始图像空间
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# 将结果图像与原始图像相结合
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)
```
