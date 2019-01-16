## 霍夫变换在车道线检测中的应用

首先，我们需要对原始图像进行灰度变换、平滑滤波以及Canny边缘检测，得到一副边缘检测后的图像。
然后，我们需要对该图像进行霍夫变换从而找出真实的车道线。

刚才我们已经了解了什么是霍夫变换，我们来看一下如何将霍夫变换用于车道线检测问题。为了实现霍夫变换，我们将会用到OpenCV中`HoughLinesP`的函数，其调用方式如下：
```python
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
```
下面，我们来依次分析相关变量的含义：

1. masked_edges表示的是输入的边缘检测图像（Canny边缘检测的输出结果）。
2. $$\rho$$和$$\theta$$是我们网格在霍夫空间中的距离和角度分辨率。即在霍夫空间中，我们会沿着（θ，ρ）轴分别的网格。其中以像素为单位指定rho，以弧度为单位指定theta。


