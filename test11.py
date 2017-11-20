import tensorflow as tf
v=tf.Variable(0,dtype=tf.float32,name="v")
#在没有申明滑动平均模型时只有一个变量v,所以下面的语句智慧输出“v:0”.
for variables in tf.global_variables():
    print(variables.name)
ema=tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op=ema.apply(tf.global_variables())
#在申明滑动平均模型之后，TensorFlow会自动生成一个影子变量
#v/ExponentialMoving Average.于是下面的语句输出
#“v:0”和“v/ExponentialMovingAverage：0”
for variables in tf.global_variables():
    print(variables.name)
saver=tf.train.Saver()
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v,10))
    sess.run(maintain_averages_op)
    #保存时，TensorFlow会将v:0和v/ExponentialMovingAverage:0两个变量都存在下来
    saver.save(sess,"model/model.ckpt")
    print(sess.run([v,ema.average(v)]))
v=tf.Variable(0,dtype=tf.float32,name="v")
#通过变量重命名将原来变量v的滑动平均值直接赋值给v.
saver=tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
    saver.restore(sess,"model/model.ckpt")
    print(sess.run(v))#输出0.099999905,这个值就是原来模型中变量v的滑动平均值


