from tensorflow.examples.tutorials.mnist import  input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
print(mnist.train)
#我们通过操作符号变量来描述这些可交互的操作单元，可以用下面的方式创建一个:
x=tf.placeholder(tf.float32,[None,784])
#x不是一个特定的值，而是一个占位符placeholder,我们在TensorFlow运行计算时输入这个值。我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None,784]
#这里的None表示此张量的第一个维度可以是任何长度的
#我们的模型也需要权重值和偏置值,当然我们可以把他们当做是另外的输入（使用占位符），但TensorFlow有一个更好的方法来表示它们:Variable。一个Variable代表一个可修改的张量,存在在TensorFlow的用于描述交互性操作的图中.它们可以用于计算的输入值，也可以在计算中被修改。对于各种机器学习应用，一般都会有模型参数，可以用Variable表示.
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
#我们赋予tf.Variable不同的初值来创建不同的Variable:在这里，我们都用全为0的张量来初始化W和b。因为要学习W和b的值，他们的初值可以随意设置.
y=tf.nn.softmax(tf.matmul(x,W)+b)
#首先，我们用tf.matmul(x,W)表示x乘以W，对应之前等式里面的Wx,这里x是一个
print ("Training data size:",mnist.train.num_examples)
print("Validating data size:",mnist.validation.num_examples)
print("Testing data size:",mnist.test.num_examples)
print("Example training data:",mnist.train.images[0])
print("Example training data label:",mnist.train.labels[0])
#为了方便使梯度随机下降input_data.read_data_sets函数生成的类还提供了mnist.train.next_batch函数，他可以从所有的训练数据读取一小部分作为一个训练batch。
batch_size=100
xs,ys=mnist.train.next_batch(batch_size)
#从train的集合中选取batch_size个训练数据
print("X shape:",xs.shape)
#输出Xshape:(100,784)
print("Y shape:",ys.shape)
#输出Y shape:(100,10)

