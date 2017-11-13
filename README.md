# tianchi_guangdong_unet

针对高分辨率遥感卫星进行地物识别，可以做二分类或者多分类问题。针对天池广东遥感大赛进行了修改，用来识别建筑物。利用Unet结构进行语义分割，得到各个地物类型的场景分割图像，Unet结构和官方论文不太一样，自己根据理解进行了一些微调，改变了输出通道的数量，和上采样层后通道数量，每个巻积层后面加了batchNromalize层，并让学习率随着epoch的增加而减小。
  
数据集：天池广东遥感比赛15和17年的4通道遥感图片，训练集使用160大小的3通道图片。

代码：基于Unet的网络结构，使用tensorflow框架实现。

训练可视化：使用tensorboard可视化训练过程，命令行运行 tensorboard --logdir="logs(使用filewriter的存储路径)"

![训练可视化](https://gitee.com/uploads/images/2017/1113/131248_a541b442_1340099.png "clipboard.png")