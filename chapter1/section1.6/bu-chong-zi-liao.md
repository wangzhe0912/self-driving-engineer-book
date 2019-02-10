## 补充资料

接下来，我们将会推荐一些相关的论文供参考，这些论文可以帮助你更好的理解计算机视觉相关的知识。


### 车道线语义分割

[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211) E. Shelhamer, J. Long and T. Darrell

摘要：
卷积网络是一种功能强大的可视化模型，可以生成特征层次结构。我们证明了通过端到端、像素到像素的卷积网络在语义分割上可以得到更好的效果。我们的核心思想是建立“完全卷积”的网络，它接受任意大小的输入，并通过有效的推理和学习产生相应大小的输出。我们定义并详细描述了完全卷积网络的空间，解释了它们在空间密集预测任务中的应用，并与先前的模型建立了联系。我们将目前的分类网络（Alexnet、VGG网络和Googlenet）调整为完全卷积网络，并通过微调将其学习到的表示转移到分割任务中。此外，我们定义了一个Skip的结构，它将来自深层、粗层的语义信息与来自浅层、细层的外观信息结合起来，以生成准确和详细的分段。

我们可以使用[KITTI road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php)数据集来建立我们的模型。



[Lane Detection with Deep Learning (Part 1)](https://towardsdatascience.com/lane-detection-with-deep-learning-part-1-9e096f3320b7)
[Lane Detection with Deep Learning (Part 2)](https://towardsdatascience.com/lane-detection-with-deep-learning-part-2-3ba559b5c5af)

优达学城的学生使用深度学习的方法进行车道线检测，最终建立了一个全卷积神经网络的模型，该神经网络可适用于各种情况的道路并且具备更快的计算速度。


### 其他的车道线检测方法

[VPGNet: Vanishing Point Guided Network for Lane and Road Marking Detection and Recognition](https://arxiv.org/abs/1710.06288)S. Lee, et. al.

摘要：
本文提出了一个统一的端到端可训练的多任务网络，该网络能够在恶劣天气条件下导致丢失点情况下的车道线和道路标记检测与识别。我们解决了夜间的雨天和低照明条件，在有限的照明条件下会发生色彩失真。此外，我们不需要大量的基准数据集就可以在恶劣天气条件下工作。我们建立了一个车道和道路标记基准，它由大约20000张图像组成，在4种不同的场景下，17个车道和道路标记类：无雨、无雨、大雨和夜间。我们对多任务网络的几个版本进行了训练和评估，并验证了每个任务下的可靠性。总结一下，VPGNet网络可以检测和分类车道和道路标记，并通过一次向前传球预测消失点。实验结果表明，该方法在各种条件下均能实时（20fps）获得较高的精度和鲁棒性。


### 车辆检测

[Learning to Map Vehicles into Bird's Eye View](https://arxiv.org/abs/1706.08442)A. Palazzi, et. al.

摘要：



你可能已经注意到上面的许多论文都使用了深度学习技术。目前，这些技术现在在许多计算机视觉应用程序中得到了普遍使用。在下一章节中，我们将会开始介绍深度学习相关的技术！
