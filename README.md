# Pattern-Recognition-Final-Homework

* Work by Lin, Tzu-Heng
* Email: lzhbrian@gmail.com


	My Java Homework for Pattern Recognition Course
	-- Make a DBN to classify a set of test data similar to MNIST but 32*32
	Using the framework of DeepLearning4j

**_Website of [DeepLeanring4j](http://deeplearning4j.org)_**

## Log
	2016.5.1 - 2016.6.20 Testing tools & Reading papers,codes
	2016.6.22 17:24 Finally understood the structure. Built a structure of DBN.
	2016.6.23 12:52 Sucking binary RBM layers.

## 报告

##### Overall:

* 这个大作业,我使用了DeepLearning4j（以下简称DL4J）的库,在IntelliJ IDEA CE的IDE上进行编程测试。
* 其实从第八周后我就开始写这个大作业了,首先遇到的问题就是要选哪一个库,一开始
* 后来,因为一些别的作业以及在做一些大数据方面的研究,Python以及Java运用得比较熟练了,所以就开始

##### Choice of Tools:

    我也尝试使用了Python的Theano,以及SciPy＋Opencv的组合等,想要来实现;
    但相较于DL4J,上述的这些库有的比较麻烦,有的功能较局限,因此最后还是选择了DL4J。

Website of Resources I have visited:
1. [DeepLeanring4j (DL4J)](http://deeplearning4j.org) (Java)
2. [DL4J: A really brief intoduction about how to use DBN on a MNIST](http://deeplearning4j.org/deepbeliefnetwork.html) (Java)
3. [DL4J: Brief instructions about how to use DBN on Iris DB](http://deeplearning4j.org/iris-flower-dataset-tutorial) (Java)
3. [Theano: Guide for RBM](http://deeplearning.net/tutorial/rbm.html) (Python)
4. [SciPy: Official Tutorial for RBM + Mnist](http://scikit-learn.org/dev/auto_examples/neural_networks/plot_rbm_logistic_classification.html) (Python)
5. [SciPy + Opencv: Tutorial MNIST by pyimagesearch.com](http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/) (Python)

Comments:
1. 上面第二个DL4J的DBN用在MNIST上的教程将MNIST先二值化{0,1},然后用RBM进行pretrain、backprop,我尝试了之后发现效果非常非常差劲;当然,他在里面也有说,里面可能要调一些参数,但是我调了非常非常久还是没什么效果。
2. 第3个DL4J构建DBN用在Iris上的教程

Databases:
1. [MNIST database](http://yann.lecun.com/exdb/mnist/)

##### Hardworking Process:

    因为网络上的内容非常良莠不齐（有些东西真心不能上网找资料…）,导致我进入了非常多的误区,于是我花了非常久的时间才完全了解RBM的全部内容,
    所以我下面先给出一个Hinton这篇论文,以及RBM一个比较完整的介绍,让以后想要用RBM、DBN的人少走一些弯路...

##### Configure a DBN (Stacked by RBMs)

    我先尝试使用了binary即二值的DBN网络，二值的方法是使用DL4j自带的函数dataSetlterator，
    如下，第三个参数设为true即表示将Mnist的图片进行二值化，灰度值大于35的即表示成1，小于35的表示成0
``DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples,true);``

