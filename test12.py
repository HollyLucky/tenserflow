import tensorflow as tf
v=tf.Variable(0,dtype=tf.float32,name="v")
ema=tf.train.ExponentialMovingAverage(0.99)
#通过使用variables_to_restore函数可以直接生成上面代码中提供的字典
#{"v/ExponentialMovingAverage":v}
#以下代码会输出：
#{'v/ExponentialMovingAverage':<tensorflow.python.ops.variables.Variable
#object at 0x7ff6454ddc10 >}
#其中后面的Variable类就代表了变量v
print(ema.variables_to_restore())
saver=tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,"model/model.ckpt")
    print(sess.run(v))#输出0.099999905，即原来模型中变量v的滑动平均值
