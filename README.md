# Pattern Recognition Work
-- For a dataset similar to MNIST

* Work by Lin, Tzu-Heng, Dept. of Electronic Engineering, Tsinghua University
* Email: lzhbrian@gmail.com
* 电子工程系 无42 林子恒 2014011054


* My Java Homework for Pattern Recognition Course
-- Make a DBN to classify a set of test data similar to MNIST but 32x32
* Using the framework of **_[DeepLeanring4j](http://deeplearning4j.org)_**


### Log
	2016.5.1 - 2016.6.20 Testing tools & Reading papers,codes
	2016.6.22 17:24 Finally understood the structure. Built a structure of DBN.
	2016.6.23 12:52 Sucking binary RBM layers.


# 报告 Reports


### 目录:
***
1. 写在前面 overall
2. 浅显易懂的解释 Introduction of RBM、DBN、AutoEncoder、Finetuning, etc
3. 库的介绍与选择 Choice of Tools
4. 我的工作 Configure a DBN






### 1. 写在前面 Overall:
***
* 这个大作业,我使用了DeepLearning4j（以下简称DL4J）的库,在IntelliJ IDEA CE的IDE上进行编程。



* About
* 其实从第八周后我就开始写这个大作业了,首先遇到的问题就是要选哪一个库,那个时候Python和Java都不太熟练,只会用Matlab。后来,因为一些别的作业以及在做一些大数据方面的研究,Python以及Java运用得比较熟练了,所以后来就使用了Theano与DL4J; (同时,也切身感受到了Matlab的鸡肋,效率实在是太低了。)



* 关于网络上的教程:
* 再后来看了很多很多RBM相关的教程,网络上的教程真心都非常非常文言文,而Hinton的那篇论文如果直接那来看的话其实一般人看不太懂(比如我)...所以我的撞墙期非常非常久,直到考完试我才比较能够理解RBM的概念。



* 关于本篇报告:
* 所以接下来我会先介绍以下Hinton在他的那篇['Reducing the Dimensionality of Data with Neural Network'](http://science.sciencemag.org/content/313/5786/504)论文(以下简称"文章")里面所说的一些内容,用正常人都能看懂的话...希望之后的同学可以不要被网上的教程迷惑...再接下来,我会给出网上一些库,包括Matlab、Python、Java的RBM教程链接,给出一些个人的比较浅显的理解。最后给出我使用DL4J来分类老师给的32x32图片的具体过程。我还是使用了MNIST数据集来训练,最后把测试集的图片剪切掉外面的两筐成28x28的来分类。




* 由于本人的水平有限,还望老师、各位指出我这篇报告里的不当之处,谢谢!






### 2. 浅显易懂的解释RBM、DBN:
***
Introduction of RBM、DBN、AutoEncoder、Finetuning, etc

##### 受限波尔兹曼机 RBM (Restricted Boltmann Machine) :
* 其实RBM就是一个两层的层内没有互相连接,层间所有都链接的一个二部图(如下图),v对应的这层称visible layer, h对应的这层称hidden layer

![pic From DL4J](http://deeplearning4j.org/img/sym_bipartite_graph_RBM.png)
(图片来自DL4J)
* Hinton在[文章](http://science.sciencemag.org/content/313/5786/504)中指出了一个能量函数,每一张图片都对应了一个能量。
* 简单来说,训练一个RBM(无监督学习),就是要使得这个RBM接收到图片之后对应的能量函数达到最小。那么训练 这个RBM有什么用呢?不要着急。


##### 深度置信网 DBN (Deep Believe Network) :
* 所谓的DBN就是将几层RBM网络堆叠(Stack)起来,下层的hiddenLayer等于上层的visibleLayer,这样就可以形成一个多层神经网络(Neural Network)


##### 预训练-调整 Pretrain (Initialize a good initial weight) - Fine-tuning(Backpropagation):
* Hinton在[文章](http://science.sciencemag.org/content/313/5786/504)中也提到,多层神经网络有一个很本质的问题就在于使用如后向传播算法(backpropagation)来迭代网络时很依赖于初始值的选择,非常容易就掉入到局部极小值的点。
* 其实整篇文章的重点就在于:
    1. 先pretrain: 将一个DBN pretrain出来一个很好的初始值,以避免掉入局部极小;
    2. 再进行Finetuning, 即后向传播算法。


##### 自编码器 AutoEncoder:



### 3. 库的介绍与选择 Choice of Tools:
***

* 我也尝试使用了Python的Theano,以及SciPy＋Opencv的组合等;
* 但相较于DL4J,上述的这些库有的比较麻烦,有的功能较局限,因此最后还是选择了DL4J。
* 下面是一些我参考过的库以及教程的链接,我接下来对他们做一些我自己的比较粗浅的理解,如有不妥之处,还望大家指正。


* Website of Resources I have visited:
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

    我先尝试使用了binary即二值的DBN网络，二值的方法是使用DL4j自带的函数dataSetlterator，
    如下，第三个参数设为true即表示将Mnist的图片进行二值化，灰度值大于35的即表示成1，小于35的表示成0
        DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples,true);

