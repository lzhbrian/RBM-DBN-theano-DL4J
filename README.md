# Pattern Recognition Work
-- For a dataset similar to MNIST

* Work by Lin, Tzu-Heng, Dept. of Electronic Engineering, Tsinghua University
    * Email: lzhbrian@gmail.com
    * 电子工程系 无42 林子恒 2014011054


* My Java Homework for Pattern Recognition Course
-- Make a DBN to classify a set of test data similar to MNIST but 32x32
    * Using the framework of **_[DeepLeanring4j](http://deeplearning4j.org)_**


### Log

	2016.5.1 - 2016.6.20 Testing tools & Reading papers, codes, etc
	2016.6.22 17:24 Finally understood the structure. Built a structure of DBN.
	2016.6.23 12:52 Sucking binary RBM layers.


# 报告 Reports


### 0. 目录:
***
0. 目录
1. 写在前面 overall
    1. About
    2. 关于网络上的教程
    3. 关于本篇报告
2. 浅显易懂的解释 Introduction of RBM、DBN、Finetuning, etc
    1. 受限波尔兹曼机 RBM
    2. 深度置信网 DBN
    3. 预训练-调整 Pretrain - Finetuning
3. 库的介绍与选择 Choice of Tools
    1. Website of Resources I have used
    2. Comments
4. 我的工作 Configure a DBN
    1. 前期准备: 编程环境、安装库
    2. 一些代码的解释
        1. 如何构造一个DBN、DBN的参数含义
        2. 读入老师提供的手写集数据
    3. 训练效果
    4. 结论
5. 写在后面:
    1. 关于我, 关于模式识别课程 About me, About Pattern Recognition





### 1. 写在前面 Overall:
***
* 这个大作业,我使用了DeepLearning4j（以下简称DL4J）的库,在IntelliJ IDEA CE的IDE上进行编程。
* About
    * 其实从第八周后我就开始写这个大作业了,首先遇到的问题就是要选哪一个库,那个时候Python和Java都不太熟练,只会用Matlab。后来,因为一些别的作业以及在做一些大数据方面的研究,Python以及Java运用得比较熟练了,所以后来就使用了Theano与DL4J; (同时,也切身感受到了Matlab的鸡肋,效率实在是太低了。)
* 关于网络上的教程:
    * 再后来看了很多很多RBM相关的教程,网络上的教程真心都非常非常文言文,而Hinton的那篇论文如果直接那来看的话其实一般人看不太懂(比如我)...所以我的撞墙期非常非常久,直到考完试我才比较能够理解RBM的概念。
* 关于本篇报告:
    * 所以接下来我会先介绍以下Hinton在他的那篇['Reducing the Dimensionality of Data with Neural Network'](http://science.sciencemag.org/content/313/5786/504)论文(以下简称"文章")里面所说的一些内容,用正常人都能看懂的话...希望之后的同学可以不要被网上良莠不齐的教程迷惑...再接下来,我会给出网上一些库,包括Matlab、Python、Java的RBM教程链接,给出一些个人的比较浅显的理解。最后给出我使用DL4J来分类老师给的32x32图片的具体过程。我还是使用了MNIST数据集来训练,最后把测试集的图片剪切掉外面的两筐成28x28的来分类。
* 由于本人的水平有限,还望老师、各位指出我这篇报告里的不当之处,谢谢!






### 2. 浅显易懂的解释 RBM、DBN:
***
Introduction of RBM、DBN、Pretrain、Finetuning, etc

* 受限波尔兹曼机 RBM (Restricted Boltmann Machine) :
    * RBM就是一个两层的层内没有互相连接,层间所有都链接的一个二部图(如下图),v对应的这层称visible layer, h对应的这层称hidden layer
    * ![pic From DL4J](http://deeplearning4j.org/img/sym_bipartite_graph_RBM.png)(图片来自DL4J)
    * Hinton在[文章](http://science.sciencemag.org/content/313/5786/504)中指出了一个能量函数,每一张图片都对应了一个能量。
    * 简单来说,训练一个RBM(无监督学习),就是要使得这个RBM接收到图片之后对应的能量函数达到最小。那么训练 这个RBM有什么用呢?不要着急。


* 深度置信网 DBN (Deep Believe Network) :
    * 所谓的DBN就是将几层RBM网络堆叠(Stack)起来,下层的hiddenLayer等于上层的visibleLayer,这样就可以形成一个多层神经网络(Neural Network),训练方法其实就是从低到上一层一层的来训练RBM。


* 预训练-调整 Pretrain (Initialize a good initial weight) - Fine-tuning(Backpropagation):
    * Hinton在[文章](http://science.sciencemag.org/content/313/5786/504)中提到,多层神经网络有一个很本质的问题就在于使用如后向传播算法(backpropagation)来迭代网络时很依赖于初始值的选择,非常容易就掉入到局部极小值的点。
    * 其实整篇文章的重点就在于:
        1. 先pretrain: 将一个DBN pretrain出来一个很好的初始值,以避免掉入局部极小;
        2. 再进行Finetuning, 即后向传播算法。



### 3. 库的介绍与选择 Choice of Tools:
***

* 我尝试使用了Python的Theano,以及SciPy＋Opencv的组合等;
* 但相较于DL4J,上述的这些库有的比较麻烦,有的功能较局限,因此最后还是选择了DL4J。
* 下面是一些我参考过的库以及教程的链接,我接下来对他们做一些我自己的比较粗浅的理解,如有不妥之处,还望大家指正。


* Website of Resources I have used:
    1. [Hinton: Hinton's official for DBN + MNIST](http://www.cs.toronto.edu/%7Ehinton/MatlabForSciencePaper.html) (Matlab)
    2. [DeepLeanring4j (DL4J)](http://deeplearning4j.org) (Java)
    3. [DL4J: A really brief intoduction about how to use DBN on a MNIST](http://deeplearning4j.org/deepbeliefnetwork.html) (Java)
    4. [DL4J: Brief instructions about how to use DBN on Iris DB](http://deeplearning4j.org/iris-flower-dataset-tutorial) (Java)
    5. [Theano: Guide for RBM](http://deeplearning.net/tutorial/rbm.html) (Python)
    6. [SciPy: Official Tutorial for RBM + Mnist](http://scikit-learn.org/dev/auto_examples/neural_networks/plot_rbm_logistic_classification.html) (Python)
    7. [SciPy + Opencv: Tutorial MNIST by pyimagesearch.com](http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/) (Python)


* Comments:
    1. 第1个库就是鼎鼎大名的深度学习鼻祖Hinton的Matlab库
    2. 上面第3个DL4J的DBN用在MNIST上的教程将MNIST先二值化{0,1},然后用RBM进行pretrain、backprop,我尝试了之后发现效果非常非常差劲;当然,他在里面也有说,里面可能要调一些参数,但是我调了非常非常久还是没什么效果。
    3. 第4个DL4J构建DBN用在Iris上的教程,与第3个不同,使用的是连续数值的RBM,训练的样本比较小,但构建网络的过程还是非常有参考价值的。
    4. 第5个Python的Theano库,里面的RBM构建起来稍微比较复杂一些。
    5. 第6,第7个使用了SciPy里的RBM;都是再Pretrain完之后再使用Logistic Regression来进行回归、分类。Scipy里目前好像还不支持连续值的RBM,只有离散值的BernouliRBM这个二值的可以用。


* Databases:
    1. [MNIST database](http://yann.lecun.com/exdb/mnist/)

### 4. 我的工作 Configure a DBN:
***
由于我主要使用的是DL4J,其他的一些库的具体操作就不在这里写出来了,在我提供的上述网址里都可以找到还不错的教程!


* 前期准备: 编译环境、安装库
    * 首先先按照了DL4J的官方[QUICKSTART](http://deeplearning4j.org/quickstart)网站,搭建好了框架,需要安装的工具有git、mvn、java等,官方推荐使用的IDE是IntelliJ IDEA CE,我现在在写的这个.md也是用这个IDE编辑的,个人感觉其实用起来还蛮舒服的。



* 一些代码的解释
    1. 如何构造一个DBN、DBN的参数含义
    2. 读入老师提供的手写集数据



* 训练结果
    * 我先尝试使用了binary即二值的DBN网络，如下，第三个参数设为true即表示将Mnist的图片进行二值化，灰度值大于35的即表示成1，小于35的表示成0:

        ```DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples,true);```

        但是训练出来的结果非常非常差劲,在pretrain的过程中,甚至误差越训练越大,具体的原因我尝试了非常非常久来找,但还是找不出来。最后的分类结果如下图,我甚至怀疑是DL4J本身的库里的RBM函数出了问题,因为使用别种layer来训练,同样的训练、测试集、迭代次数,都能有还不错的结果。

    *


* 结论



### 5. 关于我, 关于模式识别课程
***
About me, About Pattern Recognition


* 关于我:
    * 之前的学期都一直忙碌于各种各样的社工、比赛,没有把心思放在学习上,直到这个学期才后知后觉其实在大学里面学习知识才是最根本的东西,特别是如我在电子工程系,或者贵系。
    * 之前就一直对深度学习、人工智能这一块非常非常感兴趣,却一直没有一个很好的渠道来入门;我本身非常非常喜欢EECS里偏软的这一块,所以一直想转系转来贵系,因为电子系学的一些课程实在是太硬了(硬到我开始怀疑人生了...)...但后来因为觉得成本还是太高就作罢了,而且毕竟在我们系也有做这个方向的研究,或者研究生阶段在转其实也不迟!
    * 上学期在选课的时候就有特别留意计算机系的课程,抢到了一门并行计算基础、还有就是这门模式识别,前者后来被我中期退课退掉了,因为学的东西还是比较艰深苦涩、作业也有些无从下手;后者,也就是模式识别,我非常非常喜欢,这门课也是上大学以来第一门我自己想选的任选课。



* 关于模式识别课程:
    * 虽然还只是大二、还没修概率论,后来的一些课程在课堂上时有一些跟不上,但我在课后自己读Duda的Pattern Classification还算能读懂。总之,在模式识别的这门课上我学到了非常非常多有用的东西、学习方法;同时,有几次的Matlab作业还是花了我非常多的时间的...
    * 人工智能是21世纪的一个前沿课题,现在好多好多的各种应用都在与人工智能扯上关系,而模式识别,可以说是作为人工智能的一个最基础的学科了。
    * 这个学期我也参加了一个SRT项目,关于大数据的电商分析。前阵子与一名我们系的博士生以及本科生co-author了一篇paper,现在正在投稿到Ubicomp,不知道能否被接受;但不得不说的是,我在分析数据时使用的一些方法就是在课程当中学到的内容,学以致用的感觉让我非常满足。
    * 与老师相处的这一个学期非常愉快,希望以后还有机会上老师的课!
    * 最后,谢谢老师、助教这一学期来的帮助!




