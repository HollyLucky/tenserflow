import tensorflow as tf
# node1=tf.constant(3.0,dtype=tf.float32)
# node2=tf.constant(4.0)
# print(node1,node2)
# sess=tf.Session()
# print(sess.run([node1,node2]))
# node3=tf.add(node1,node2)
# print("node3:",node3)
# print("sess.run(node3)",sess.run(node3))
# a=tf.placeholder(tf.float32)
# b=tf.placeholder(tf.float32)
# adder_node=a+b
# print(sess.run(adder_node,{a:3,b:4}))
# print(sess.run(adder_node,{a:[1,2],b:[3,4]}))
# add_and_triple=adder_node*3
# print(sess.run(add_and_triple,{a:3,b:4}))
# W=tf.Variable([.3],dtype=tf.float32)
# b=tf.Variable([-.3],dtype=tf.float32)
# x=tf.placeholder(tf.float32)
# linear_model=W*x+b
# init=tf.global_variables_initializer()
# sess.run(init)
# W=tf.Variable([.3],dtype=tf.float32)
# b=tf.Variable([-.3],dtype=tf.float32)
# x=tf.placeholder(tf.float32)
# linear_model=W*x+b
# init=tf.global_variables_initializer()
# sess.run(init)
# print(sess.run(linear_model,{x:[1,2,3,4]}))
# y=tf.placeholder(tf.float32)
# sqared_deltas=tf.square(linear_model-y)
# loss=tf.reduce_sum(sqared_deltas)
# print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
# fixW=tf.assign(W,[-1.])
# fixb=tf.assign(b,[1.])
# sess.run([fixW,fixb])
# print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
# optimizer=tf.train.GradientDescentOptimizer(0.01)
# train=optimizer.minimize(loss)
# sess.run(init)
# for i in range(1000):
#     sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
# print(sess.run([W,b]))
# a=tf.constant([1.0,2.0],name="a")
# b=tf.constant([2.0,3.0],name="b")
# result=tf.add(a,b,name="add")
# print(result)
# a=tf.constant([1,2],name="a")
# b=tf.constant([2.0,3.0],name="b")
# result=tf.add(a,b,name="add")
# print(result)
# sess=tf.Session()
# a=tf.constant([1,2],name="a")
# b=tf.constant([3,4],name="b")
# result=tf.add(a,b,name="add")
# # print(sess.run(result))
# print(result.eval(session=sess))
# weights=tf.Variable(tf.random_normal([2,3],stddev=2))
# biases=tf.Variable(tf.zeros([3]))
# w2=tf.Variable(weights.initial_value())
# w3=tf.Variable(weights.initial_value()*2.0)
# w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
# x=tf.constant([[0.7,0.9]])
# a=tf.matmul(x,w1)
# y=tf.matmul(a,w2)
# sess=tf.Session()
# init_op=tf.global_variables_initializer()
# sess.run(init_op)
# print(sess.run(y))
# sess.close()
# w3=tf.Variable(tf.random_normal([2,3],stddev=1),name="w3")
# w4=tf.Variable(tf.random_normal([2,2],stddev=1),name="w4")
# tf.assign(w3,w4,validate_shape=False)
# w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
# x=tf.placeholder(tf.float32,shape=(3,2),name="input")
# a=tf.matmul(x,w1)
# y=tf.matmul(a,w2)
# sess=tf.Session()
# init_op=tf.global_variables_initializer()
# sess.run(init_op)
# print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
# cross_entropy=-tf.reduce_mean(y*tf.log(tf.clip_by_value(y,1e-10,1.0)))
# learning_rate=0.001
# train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# sess.run(train_step)
# from numpy.random import RandomState
# batch_size=8
# w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
# x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
# y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')
# a=tf.matmul(x,w1)
# y=tf.matmul(a,w2)
# cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1)))
# train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# rdm=RandomState(1)
# dataset_size=128
# X=rdm.rand(dataset_size,2)
# Y=[[int(x1+x2<1)] for x1,x2 in X]
# with tf.Session() as sess:
#     init_op=tf.global_variables_initializer()
#     sess.run(init_op)
#     print(sess.run(w1))
#     print(sess.run(w2))
#     STEPS=5000
#     for i in range(STEPS):
#         start=(i*batch_size)% dataset_size
#         end=min(start+batch_size,dataset_size)
#         sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
#         if i%1000==0:
#             total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
#             print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))
#     print(sess.run(w1))
#     print(sess.run(w2))
# v=tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
# sess=tf.Session()
# print(tf.clip_by_value(v,2.5,4.5).eval(session=sess))
# v=tf.constant([1.0,2.0,3.0])
# sess=tf.Session()
# print(tf.log(v).eval(session=sess))
# v1=tf.constant([[1.0,2.0],[3.0,4.0]])
# v2=tf.constant([[5.0,6.0],[7.0,8.0]])
# sess=tf.Session()
# print(sess.run(v1*v2))
# print(sess.run(tf.matmul(v1,v2)))
# v=tf.constant([[1.0,2.0,3.0],[4.0,5.,6.]])
# sess=tf.Session()
# print(sess.run(tf.reduce_mean(v)))
# v1=tf.constant([1.0,2.0,3.0,4.0])
# v2=tf.constant([4.0,3.0,2.0,1.0])
# sess=tf.InteractiveSession()
# loss=tf.reduce_sum(tf.where(tf.greater(v1,v2),v1,v2))
# print(sess.run(loss))
# sess.close()
# from numpy.random import RandomState
# batch_size=8
# x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
# y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')
# w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
# y=tf.matmul(x,w1)
# loss_less=10
# loss_more=1
# loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))
# train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
# rdm=RandomState(1)
# data_size=128
# X=rdm.rand(data_size,2)
# Y=[[x1+x2+rdm.rand()/10.-0.05] for(x1,x2) in X]
# with tf.Session() as sess:
#     init_op=tf.global_variables_initializer()
#     sess.run(init_op)
#     STEPS=5000
#     for i in range(STEPS):
#         start=(i*batch_size)%data_size
#         end=min(start+batch_size,data_size)
#         sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
#     print(sess.run(w1))
# global_step=tf.Variable(0)
# #通过exponential_decay函数生成学习率
# learning_rate=tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
# #使用指数衰减的学习率。在minimize函数中传入global_step将自动更新,global_step参数，从而使得学习率也得到相应更新
# learning_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
# w=tf.constant(tf.random_normal([2,1],stddev=1,seed=1))
# y=tf.matmul(x,w)
# loss=tf.reduce_mean(tf.square(y_-y))+tf.contrib.layers.12_regularizer(lambda)(w)
# weights=tf.constant([[1.0,-2.0],[-3.0,4.0]])
# with tf.Session() as sess:
#     print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
#     print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))
# def get_weight(shape,lambdal):
#     var=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
#     tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambdal)(var))
#     return var
# x=tf.placeholder(tf.float32,shape=(None,2))
# y_=tf.placeholder(tf.float32,shape=(None,1))
# batch_size=8
# layer_dimension=[2,10,10,10,2]
# n_layers=len(layer_dimension)
# cur_layer=x
# in_dimension=layer_dimension[0]
# for i in range(1,n_layers):
#     out_dimension=layer_dimension[i]
#     weight=get_weight([in_dimension,out_dimension],0.001)
#     bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))
#     cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
#     in_dimension=layer_dimension[i]
# mse_loss=tf.reduce_mean(tf.square(y_-cur_layer))
# tf.add_to_collection('losses',mse_loss)
# loss=tf.add_n(tf.get_collection('losses'))


