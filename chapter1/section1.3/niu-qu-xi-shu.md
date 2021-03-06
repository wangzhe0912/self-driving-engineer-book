## 图像校准

图像失真是指由于摄像机误差导致拍摄出来的图像与真实世界存在一定的扭曲、尤其是在边缘附近则更加明显，可能会存在不同程度的拉伸或倾斜。而**图像校准**则是表示通过一系列方法，将将摄像机得到的失真图像恢复。

图像失真往往是由于摄像头将真实三维世界拍摄为二维平面图像时引用的偏差。

例如，上图为原始真实世界的场景，下方的三幅图像表示由不同的摄像头拍摄后得到的一些轻微失真的图像。
![失真图像](/assets/33.jpg)

在这些失真的图像中，我们可以看到车道线的边缘发生了一定程度的扭曲，从原来的直线变为了带有一个弧度的曲线。同时，图像的失真在一定程度上也改变了物体的形状和大小。这对于我们期望精确定位无人车的位置以及识别车道线和障碍物都带来了较大的影响。因此，接下来，我们将要解决的问题就是消除这种图像失真，即**图像校准**。

Question：图像失真会带来哪些影响？

1. 改变图像中物体的形状。
2. 改变图像中物体的大小。
3. 改变图像中物体距离我们位置的远近。
4. 导致观察到的图像中的物体外型依赖于拍摄的视角。

### 针孔照相机模型

在开始学习图像校准之前，我们首先来看下图像的失真是如何发生的。下图是一个针孔照相机的基本模型：
![针孔照相机](/assets/34.jpg)
其观察世界的方式与人眼类似，即通过聚焦物体反射的光呈现得到的。
对于一个真实世界中的三维物体，通过映射后变为了一个二维平面图像，这个转换是通过一个转换矩阵完成的，我们称之为相机矩阵。

对于真实照相机而言，其使用的并不是针孔，而是镜头。光线经过镜头时，会产生一定幅度的弯曲，从而导致我们之前所观察到的图像扭曲。这种失真称之为**径向失真**，也是最常见的失真类型。

另外一种失真类型我们称之为**切向畸变**。当摄像机镜头与摄像机胶片或传感器所在的成像平面不平行时，会导致所谓的切向畸变。此时，会导致物理看起来比真实情况显得更近或更远。如下图所示：

![切向畸变](/assets/36.jpg)

幸运的是，为了消除镜头的径向失真和切向畸变，我们仅仅需要确定该照相机的如下5个系数即可：$$k_1, k_2, p_1, p_2, k_3$$。

当我们获得到该照相机的这5个扭曲相关的系数，我们就可以利用相关公式对照相机得到的图像进行还原。

![图像校准](/assets/37.jpg)

### 失真系数及其校准

关于径向失真，存在如下三个失真系数：$$k_1, k_2, k_3$$。为了校准由于径向失真导致的图像扭曲，我们可以使用根据如下失真公式进行推到处理：

$$
x = x_{corrected}(1 + k_1r^2 + k_2r^4 + k_3r^6)
$$
$$
y = y_{corrected}(1 + k_1r^2 + k_2r^4 + k_3r^6)
$$

在上述公式中，x和y表示失真图像中的某一个点。x_corrected和y_corrected表示失真对象中x,y点对应于原始图像中点的坐标。x_c和y_c表示原始图像中的中心点。r表示的则是在原始图像中，x_corrected,y_corrected到x_c,y_c点的距离。如下图所示：

![计算逻辑](/assets/38.jpg)

Ps：虽然失真系数k3对于径向失真存在一定影响，然而，对于大多数普通相机镜头所具有的轻微径向失真而言，k3接近或等于零，通常可以可忽略不计。所以，在OpenCV中，我们可以选择忽略这个系数; 这也就是为什么它出现在失真值数组的末尾：[k1，k2，p1，p2，k3]。 在本课程中，我们将在所有校准计算中都使用它，以便我们的计算适用于更广泛的镜头。

同样，对于切向畸变而言，我们也可以使用如下的公式进行图像校准：

$$
x_{corrected} = x + [2p_1xy + p_2(r^2+2x^2)]
$$
$$
y_{corrected} = y + [p_1(r^2+2y^2) + 2p_2xy]
$$

**Question**：针孔摄像机模型和镜头摄像机的模型区别是什么？

A. 针孔摄像机模型的成像是翻转的，而镜像摄像机没有翻转。
B. 针孔摄像机成像是黑白的，而镜像摄像机是彩色的。
C. **针孔摄像机不存在图像失真的问题，而镜像摄像机引入和图像失真。**
D. 针孔摄像机和镜像摄像机无差异。


### 图像校准流程

Step1. 对一个已知的物体进行拍照（如棋盘）
Ps：棋盘非常适合用于校准摄像机镜头，因为它是一个非常规则且高对比度的物体，从而使得检测过程变得非常简单。
![失真检测](/assets/39.jpg)
我们知道，一个没有失真的平面棋盘如左图所示。而失真后的图像可能如右图所示。此时，我们可以用同一个相机从多个角度拍摄该图片，如下所示：
![失真图像](/assets/40.jpg)
接下来，我们可以通过对比这些图像中的差异和理想图像的差异，从而计算出失真系数。
Step2. 检测失真误差并计算失真系数
Step3. 将计算得到的失真系数用于校准后续摄像机拍摄得到的图像。

### Python实现
在接下来的练习中，我们将使用opencv中相关的函数进行图像校准。

首先，我们需要使用`findChessboardCorners()`和`drawChessboardCorners()`函数来自动查找和绘制棋盘图案图像中的角。

[findChessboardCorners()官方文档](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners)
[drawChessboardCorners()官方文档](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.drawChessboardCorners)

通过应用上述的两个函数，我们可以如下图像
![输入图像](/assets/42.jpg)
并得到一个所下图所示的图片：
![画出图像中所有的角点](/assets/41.jpg)
在该图像中，我们找出了图像中所有的角点（两个黑色和两个白色方块相交的点）。

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 计算交点数量
nx = 8 # 数出横向存在多少个内测交点
ny = 6 # 数出纵向存在多少个内测交点

# 读取原始图像
fname = 'calibration_test.png'
img = cv2.imread(fname)

# 对图像进行灰度变换
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 输入灰度图像和交点数量，用于找出交点
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# 如果找点了交点，则在图像中画出所有交点
if ret == True:
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
```

接下来，我们将学习完整的图像校准的实现。
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取棋盘的多张拍摄图像，建议使用至少20张图像
# 所有图像使用同一个照相机针对同一个棋盘，仅仅拍摄角度和距离不一致
objpoints = []   # 真实世界中的三维图像点
imgpoints = []   # 图像平台中的二维点

# 准备objpoint，例如(0,0,0)，(7,5,0)
objp = np.zeros((6 * 8, 3), np.float32)
objp[:,:,2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

# 读取准备的多张图像
images_list = ['image1.png']
for image in images_list:
    # 依次处理每张图像
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# 读取处理好的校准图像
# dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
# objpoints = dist_pickle["objpoints"]
# imgpoints = dist_pickle["imgpoints"]

# 读取一张检测图像，计算失真系数并用于图像校准并检测校准效果
img = cv2.imread('test_image.png')

def cal_undistort(img, objpoints, imgpoints):
    # 图像校准函数
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

undistorted = cal_undistort(img, objpoints, imgpoints)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```
