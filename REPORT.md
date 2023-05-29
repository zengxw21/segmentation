## 文本情感分析 实验报告
计12 曾宪伟 2021010724

#### 0.概要
实现了基于RNN和CNN的文本情感分析，其中RNN实现了双向LSTM、GRU、加入Attention机制的BiLSTM，CNN实现了textCNN；并额外实现了MLP和transformer作为baseline对照；尝试实现了bert预训练模型，但是因为显卡内存不够无法训练。

**使用方法**
在powershell终端，使用`python main.py -n [RNN_GRU | textCNN | RNN_LSTM | LSTM_Attention | MLP | transformer ] -b [batch_size] -l [learning_rate] -e [epochs] -m [max_sentence_length]`指定超参数和模型运行，或`python main.py `运行默认设置

#### 1.模型结构

**1.1 biLSTM**

双向lstm的结构如图所示：
<div align="center">
<img src="pic/bilstm.PNG" width= "400" />
</div>

1.Embedding层，首先将词映射为固定长度的向量
2.双向双层LSTM层，接受词向量构成的序列，每个LSTM单元在两个方向上产生隐藏状态，第二层的首尾两个单元将自己的隐藏状态传递到下一层
3.全连接层，接受上一层传来的两个隐藏状态拼接而成的向量，通过两层线性层得到维数等于类别数的向量

**1.2 GRU**
GRU单元结构如图所示：
<div align="center">
<img src="pic/gru.PNG" width= "400" />
</div>

将LSTM的控制单元换成GRU单元即得到GRU网络

**1.3 LSTM with Attention**
加入Attention机制的LSTM结构如图：
<div align="center">
<img src="pic/bilstm_att.PNG" width= "400" height="230" />
</div>

传统的BiLSTM中，用最后一个时序的输出向量作为特征向量，加入了Attention层后，先计算每个时序的权重，然后将所有时序的向量进行加权和作为特征向量，然后进行softmax分类。

**1.4 textCNN**
textCNN结构图：
<div align="center">
<img src="pic/text.PNG" width= "500" />
</div>

1.Embedding层，将输入的单词映射为固定维度的向量
2.一维多通道多卷积核卷积层，将嵌入层得到的数据视为一批多通道的一维张量；一维张量的长度为对齐后的句子长度，通道数为词向量数。用指定数量与大小的卷积核与输入数据做多通道多卷积核卷积，得到多通道的一维输出特征。
3.池化层，对输出做 activate, Dropout, max pooling。
4.全连接层，将池化结果拼接得到维数等于通道数的向量输入，输出维数等于分类数的向量。

**1.5 MLP (as baseline)**
<div align="center">
<img src="pic/mlp.PNG" width= "500" />
</div>

1.Embedding层，将单词映射为指定长度的向量
2.全连接层，将句子的所有词向量拼接，输入线性层1，将输出做batch norm和dropout后输入线性层2，输出维数等于类别数的向量

**1.6 transformer (as baseline)**
<div align="center">
<img src="pic/trans.PNG" width= "500" />
</div>

**1.6.1 Position Encoding** 
由于没有RNN的序列结构，所以使用Position Encoding 的方式来表示上下文关系
$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$
$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$

其中PE是一个二维矩阵，形状就是sequence_length×embedding_dim，pos是单词在句子中的位置，d_model表示词嵌入的维度，i表示词向量的位置。奇数位置使用cos，偶数位置使用sin。这样就根据不同的pos以及i便可以得到不同的位置嵌入信息，然后，PE同对应单词的embedding相加，输入给第二层。

**1.6.2. multi-attention**
通过多个self attention组合成multi-attention，每个head学习到不同的特征

**1.6.3. add&norm**
进行残差连接和layer normalization

**1.6.4. feedforward**
做两次线性变换，经过relu激活，重复add&norm

decoder和encoder的结构类似，仅在开头加了masked multi-attention确保当前位置预测只取决于前面的预测

#### 2. 模型效果比较分析
**1.除了transformer以外的五种模型的效果比较**
**train**
<div align="center">
<img src="pic/train_loss.PNG"   />
</div>
<div align="center">
<img src="pic/train_f1.PNG"   />
</div>
<div align="center">
<img src="pic/train_acc.PNG"   />
</div>

**val**
<div align="center">
<img src="pic/val_loss.PNG"   />
</div>
<div align="center">
<img src="pic/val_f1.PNG"   />
</div>
<div align="center">
<img src="pic/val_acc.PNG"   />
</div>

**test**
<div align="center">
<img src="pic/test_loss.PNG"   />
</div>
<div align="center">
<img src="pic/test_f1.PNG"   />
</div>
<div align="center">
<img src="pic/test_acc.PNG"   />
</div>


可以看出，不论是训练达到稳定时，还是在验证和测试的最好情况下，MLP的loss都比其他四种网络高，f1_score和accuracy比其他四种低。

五种网络的train_loss都在下降后趋于稳定，除了MLP以外，其他四种网络的val_loss在前期小幅度下降后开始上升，可以推断出现了过拟合。

对每个模型取其在所有epochs中、自己的所有超参数中表现最好的test_acc，有
|textCNN|RNN_LSTM|RNN_GRU|MLP|LSTM_Attention
|---|---|---|---|---|
|0.8726|0.8591|0.8537|0.8347|0.8537

多次实验发现，
1.textCNN的表现为五种模型中最优，MLP最差。
2.加入Attention机制后，对LSTM在测试集上的最优表现影响不大，但是能抵抗过拟合条件下LSTM在测试集上正确率的迅速下降
3.GRU在验证和测试集的loss上升最快，过拟合出现最快

**2. transformer效果分析**
神奇的是，在实验中，transformer表现出来的性质与其他五种模型非常不一样。
</div>
<div align="center">
<img src="pic/trans_train_acc.PNG"   />
</div>
</div>
<div align="center">
<img src="pic/trans_train_loss.PNG"   />
</div>
</div>
<div align="center">
<img src="pic/trans_train_f1.PNG"   />
</div>
</div>
<div align="center">
<img src="pic/trans_val_acc.PNG"   />
</div>
</div>
<div align="center">
<img src="pic/trans_val_loss.PNG"   />
</div>
</div>
<div align="center">
<img src="pic/trans_val_f1.PNG"   />
</div>
</div>
<div align="center">
<img src="pic/trans_test_acc.PNG"   />
</div>
</div>
<div align="center">
<img src="pic/trans_test_loss.PNG"   />
</div>
</div>
<div align="center">
<img src="pic/trans_test_f1.PNG"   />
</div>
可以看到，transformer的val loss和train loss在第6个epoch之后趋于稳定,并没有出现过拟合情况，test_acc和test_f1也在第6个epoch后趋于饱和。与此完全不同，其他五种网络的test_acc都在第2个epoch左右达到峰值，然后在接下来呈下降趋势。

transformer的在测试集上的最好表现为0.8022，比MLP baseline 还要差。推测是因为transformer需要更大量的数据集来训练，和预训练的词向量也有一定关联。


#### 3.不同参数的效果
**batch_size**
以textCNN为例，观察batch size不同导致的变化
<div align="center">
<img src="pic/batch_train_loss.PNG"   />
</div>
<div align="center">
<img src="pic/batch_val_loss.PNG"   />
</div>
<div align="center">
<img src="pic/batch_test_acc.PNG"   />
</div>
可以发现，随着batch size变大，train loss收敛的值变大，val loss收敛值变小；过大或过小的batch size均会降低模型的最优性能。

batch size太大，模型权重很难更新，梯度方向几乎没变化，容易陷入局部极小值；太小，梯度震荡剧烈；使用mini batch有利于收敛，但不一定提高模型精度。

**max sentence length**
以textCNN为例。
<div align="center">
<img src="pic/ml_test_acc.PNG"   />
</div>

多次实验发现，最大句长减小时，模型无法捕捉到全局信息，在测试集上的正确率相应的下降了


**learning rate**
以textCNN为例。
<div align="center">
<img src="pic/lr_train_loss.PNG"   />
</div>
<div align="center">
<img src="pic/lr_train_acc.PNG"   />
</div>
<div align="center">
<img src="pic/lr_val_loss.PNG"   />
</div>
<div align="center">
<img src="pic/lr_test_acc.PNG"   />
</div>
可见，学习率过小，train loss收敛值较大，train acc收敛值较小。学习率过大，虽然训练集上面的表现与最优学习率几乎无异，但是val loss同最优相比发生了爆炸（test loss情况与val loss一致，此处省略图片）。在测试集上的准确率，最优学习率（也是Adam优化器的默认学习率，大小适中）最高，大学习率次之，小学习率最差。
学习率过大，会导致网络不能收敛，在最优值附近徘徊，在此表现为test acc无明显峰值；学习率过小，虽然有可能找到最优值，但是还有可能陷入局部最小值收敛，在此表现为test acc收敛值较小

#### 4.baseline模型和CNN、RNN模型的效果差异
将 MLP 和 transformer 作为baseline，进行分析

- **train loss**
达到稳定时，
$ transformer > MLP > CNN \approx RNN $

- **val loss**
在小幅度下降后，GRU增长最快，LSTM类次之，textCNN和MLP增长最慢；transformer在下降后趋于稳定

- **test accuracy**
$ transformer < MLP < GRU \approx LSTM \approx {LSTM}_{Attention} < textCNN$

- **test f_score**
情况与test accuracy 相同

#### 5.问题思考

**1.实验训练什么时候停止是最合适的？简要陈述你的实现方式，并试分析固定迭代次数与通过验证集调整等方法的优缺点。**

当train loss下降幅度较小，基本收敛，且val loss 开始上升时，停止训练最合适，这个时候模型开始出现过拟合，而当val loss的拐点到来后的一小段时间内，test accuracy还会短暂上升，在这个区间内选取最优模型并停止训练。我在实现的时候按照上述方法做的。

固定迭代次数对transformer可能比较有效，因为它在训练集和验证集上的loss是单调递减的，但是由于数据集规模限制，其会在不大的epoch内逐渐收敛到一个固定值。此时若epoch还没结束，会浪费算力。

通过验证集调整对其他五种模型有效，可以防止因为学习能力太强出现过拟合。

**2.实验参数的初始化怎么做的？不同的方法适合哪些地方？**
Conv layer、LSTM layer、GRU layer、Linear layer使用kaiming初始化，Embedding layer、Attention layer使用均匀分布初始化。MLP使用高斯初始化

**零均值初始化**
这样获得的数据有正有负，正态分布，比较合理，但是不适用于很深的网络，因为当初始化权重较大时，每层输出的方差会越来越大，发生梯度爆炸；初始化权重较小时，每层输出的方差越来越小，最后梯度就贴近0，也就是说权值根本不会根据误差的大小而有所更新。

**正交初始化**
在RNN中用来对抗梯度消失和爆炸的方法。将权重矩阵进行特征值分解后可以发现，梯度消失源于特征值小于1的矩阵的连续乘法，梯度爆炸源于特征值大于1的矩阵的连续乘法。用SVD构建出正交矩阵，特征值全为1，用来初始化权重矩阵。在训练层数多的以及RNN网络时较有效。

**Xavier初始化**
用来解决梯度消失问题。简单来说，要解决梯度消失，就要避免激活值方差的衰减，即每一层输出的方差应该尽量相等。
Xavier初始化的基本思想是让一层网络的输入输出可以保持正态分布且方差相近。
对于线性激活函数，权重初始化满足：
$若W_{ij}服从正态分布，则W_{ij} \sim N(0,\frac{2}{d+u})$;
$若W_{ij}服从均匀分布，则W_{ij} \sim U(-\sqrt{\frac{6}{d+u}},\sqrt{\frac{6}{d+u}}) $
其中，d表示输出神经元数量，u表示输入神经元数量
因为sigmoid函数和tanh函数在0附近近似于线性函数，所以Xavier初始化效果较好。

**kaiming初始化**
解决Xavier初始化在relu函数上表现不佳的问题。Xavier推导过程假设激活函数在零点附近接近线性函数，且激活值关于0对称。Relu函数很明显是不对称的。
kaiming初始化基本思想是由于ReLU函数让一半的Z值（负值）变为零，实际上移除了大约一半的方差。所以需要加倍权重的方差以补偿这一点，也就是将权重的方差乘以2。
$若W_{ij}服从正态分布，则W_{ij} \sim N(0,\frac{2}{d})$;
$若W_{ij}服从均匀分布，则W_{ij} \sim U(-\sqrt{\frac{6}{d}},\sqrt{\frac{6}{d}}) $
其中，d 可以是输出层神经元个数，也可以是输入层神经元个数，二选一。
kaiming初始化是pytorch默认的初始化方式。

**3.过拟合是深度学习常见的问题，有什么方法可以防止训练过程陷入过拟合**
过拟合是因为是因为模型在训练中过于依赖了训练数据中的噪声和特有的特征，而无法泛化到新数据上。防止训练陷入过拟合的方法有：
1.数据增强：对训练数据进行一定程度的变换，或者扩充数据集的大小和多样性
2.正则化：通过 L1、L2 等正则化方法，对模型的权重进行约束，避免模型过拟合训练数据。
3.Dropout：在神经网络中随机地关闭一些神经元，以减少神经元之间的耦合，从而防止过拟合。
4.早停法（Early stopping）：在训练过程中，使用验证集来监测模型的泛化能力，验证集上正确率不再提升时，及时停止训练。

**4.分析CNN，RNN，MLP的优缺点**

**CNN**
- 优点：
能提取输入数据的特征；模型简单, 训练速度快，效果好
- 缺点：
模型可解释性不强；对于文本中的长距离依赖性较差，无法捕捉到全局语义信息。

**RNN**
- 优点：
能够有效地处理有序数据，并且充分利用了序列中每个元素的信息；有记忆功能，能根据历史信息进行推理预测
- 缺点：
训练难度大，计算量大，容易出现梯度消失或爆炸的问题；长时间序列的处理效果较差，因为长期记忆需要模型较大的存储空间和计算资源。

**MLP**
- 优点：
模型简单直观，训练速度快
- 缺点：
模型性能不佳，参数量大，训练难度大；容易过拟合

#### 6.心得体会
1.通过本次作业，我学会了如何使用pytorch框架完成一次有实际意义的深度学习任务，了解了深度学习中常用的评价模型性能的指标
2.在本次实验中，除了MLP外还完成了transformer和BERT，但是结果不尽如人意：transformer在小数据集文本分类上的表现明显劣于CNN、RNN，且出现了瓶颈（train loss和val loss收敛）；BERT因为参数量过于巨大，在因为显存不够而无法运行。这启示我在一些小型的下游任务中，使用过于复杂的模型有时候会适得其反，最终模型的选择应该基于实现成本、计算成本、任务规模、模型性能做综合考量。