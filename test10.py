import tensorflow as tf
#这里声明的变量名称和已经保存的模型中变量的名称不同
v1=tf.Variable(tf.constant(1.0,shape=[1]),name="other-v1")
v2=tf.Variable(tf.constant(2.0,shape=[1]),name="other-v2")
#如果直接使用tf.train.Saver()来加载模型会报变量找不到的错误。下面显示了报错信息：
#tensorflow.python.framework.errors.NotFoundError:Tensor name "other-v2"
#not found in checkpoint files/path/to/model/model.ckpt
#使用一个字典(dictionary)来重命名变量可以就可以加载到原来的模型了。这个字典指定了
#原来名称为v1的变量现在加载到变量v1中（名称为other-v1）,名称为v2的变量
#加载到变量v2中（名称为other-v2）
saver=tf.train.Saver({"v1":v1,"v2":v2})