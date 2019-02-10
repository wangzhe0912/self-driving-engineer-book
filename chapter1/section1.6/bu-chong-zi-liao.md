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
