## 机器学习基本概念

无人驾驶近年来的一些最重要的技术突破主要来自于机器学习。

**机器学习**属于人工智能领域，它让计算机通过数据而不是依赖于程序员设定的规则、条件来进行判断、计算和决策。
**深度学习**是使用深度神经网络进行机器学习的一种方法，通过深度神经网络 深度学习可以完成从语音识别到汽车驾驶等一系列复杂的任务。


在本节中，我们将会：

首先学习感知机 (perceptron) 的概念，这是神经网络的基本组成单元。
然后，将学习如何把这些感知器单元组成一个简单的神经网络。


下面，我们将通过一个基础的场景来理解一下什么是机器学习。

假设我们正在研究房地产市场，我们的任务是根据房屋面积估计它的价格。已知有一栋较小的房子售价为$70,000，一栋较大的房子售价为$160,000，那么我们希望估计这栋中等面积房子的价格 该怎么估算？

![](/assets/91.jpg)

我们先把房屋的面积和对应价格放到坐标系中，其中x轴表示以平方英尺为单位的房子面积，y轴表示价格。图中的蓝色数据点是我们收集到的房屋数据。从图中可看出这栋小房子价格为$70,000  大房子价格为$160,000。根据以上给出的数据 你认为这栋中等面积房子的最佳估价是多少？

此时，我们可以根据我们收到的数据信息在坐标系中画一条直线，该直线近似可以表示房屋面积和房价的关系曲线。此时，我们可以将中等面积房屋映射到该直线上，并找出该直线对应的y坐标。

![](/assets/92.jpg)

**显然，从图像中可以看到，中等面积的房子对应的价格最接近于$120,000。**

虽然上述过程中，我们没有提及任何的技术词汇，但在这一过程中，我们实际使用的手段称为**线性回归**。线性回归是根据你收集的数据，找出一条最拟合于所收集的数据的直线，并根据拟合的直线进行预测的方式。


### 线性回归VS逻辑回归

线性回归可以用于预测连续函数的值，例如预测房屋的价格。那么我们如何在离散序列之间对数据进行分类呢？例如：

1. 确定患者是否患有癌症
2. 识别鱼的种类
3. 识别谁在进行电话会议

分类问题对于自动驾驶汽车很重要。自动驾驶汽车需要对道路中的物体进行分类，例如是汽车，行人还是自行车。此外，自动驾驶汽车还需要确定出现哪种类型的交通标志，或者当前交通信号灯是什么颜色等等。

我们先来看一个分类的例子，假设我们是一所高校的招生人员。我们的工作就是接受或拒绝申请的学生。

我们可以利用两方面的信息来评估这些学生。分别是：考试成绩和在校期间的平时成绩。如下为一些示例学生的信息：

1. 学生1的考试得分是9分（满分10分）平时成绩为8分（满分10分）。结果：录取
2. 学生2考试得分是3分（满分10分）平时成绩为4分（满分10分）。结果：未录取
3. 学生XX考试得了XX分，平时成绩XX分。结果XX。（省略，不一一列举）
4. 学生n考试得了7分（满分10分）平时成绩6分（满分10分），我们期望预测该学生是否被录取。


下面，我们需要在图表中表示学生们的成绩，其中，横轴表示考试得分、纵轴则表示平时成绩，则每个学生对应于坐标轴中的一个点。此外，红色点表示该学生未被录取，绿色点则表示该学生被录取。则样本结果在坐标轴中的可视化如下：

![](/assets/93.jpg)

现在，我们再来预测下学生n考试得了7分（满分10分）平时成绩6分（满分10分）是否会被录取吧。

![](/assets/94.jpg)

与之前线性回归的方法类似，我们在坐标轴中找出一条直线，近似可以将录取和未录用的学生进行分类。此时，位于直线上方的大多数学生都被录取了，而位于直线下方的大多数学生则被拒绝。所以可以把这条直线作为我们的模型进行后续数据的预测。基于这个模型，我们可以看到学生n（7,6）位于直线上方，所以我们可以比较确定这个学生会被录取。

那么，问题来了，我们应该如何让计算机能够找出这条直线呢？这就是机器学习所要解决的核心问题。这将是我们接下来要讨论的内容。

**分界线表示**

我们将上述坐标轴中的横坐标轴称为x1，纵坐标轴成为x2。此时，这条分隔蓝点和红点的分界线可以用线性方程表示：

$$
2x_1+x_2-18=0
$$

此时，我们可以将x1,x2作为输入，$$2x_1+x_2-18$$的计算结果作为输出，当其大于0时，表示该学生会被录取，当其小于0时，则表示该学生不会被录取。
























