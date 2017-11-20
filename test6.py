import tensorflow as tf
v=tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))
v=tf.Variable(tf.constant(1.0,shape=[1]),name="v")
with tf.variable_scope("foo"):
    v=tf.get_variable("v",[1],initializer=tf.constant_initializer(1.0))
#因为在命名空间foo中已经存在名字为v的变量，所有下面的代码将会报错：
#Variable foo/v already exists disallowed.Did you mean to set reuse=True
#in VarScope?
# with tf.variable_scope("foo"):
#     v=tf.get_variable("v",[1])
#在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable函数直接获取已经声明的变量
# with tf.variable_scope("foo",reuse=True):
#     v1=tf.get_variable("v",[1])
#     print(v1==v)
# #将参数reuse设置为True时，tf.variable_scope将只能获取已经创建过的变量。
# #因为在命名空间bar中还没有创建变量v，所以下面的代码将会报错：
# # with tf.variable_scope("bar",reuse=True):
# #     v=tf.get_variable("v",[1])
# with tf.variable_scope("root"):
#     #可以通过tf.get_variable_scope().reuse函数来获取当前上下文管理器中reuse参数的取值
#     print(tf.get_variable_scope().reuse)
#     with tf.variable_scope("foo",reuse=True):#新建一个嵌套的上下文管理器并指定reuse为True
#         print(tf.get_variable_scope().reuse)#输出True
#         with tf.variable_scope("bar"):#新建一个嵌套的上下文管理器但不指定reuse，这时reuse的取值会和外面一层保持一致
#             print(tf.get_variable_scope().reuse)#输出True
#     print(tf.get_variable_scope().reuse)#输出False
# v2=tf.get_variable("v2",[1])
# print(v2.name)#输出v:0.“v”为变量的名称,“：0”表示这个变量生成变量这个运算的第一个结果
# with tf.variable_scope("foo"):
#     v3=tf.get_variable("v3",[1])
#     print(v3.name)#输出foo/v:0.在tf.variable_scope中创建的变量，名称前面会加入命名空间的名称，并通过/来分割命名的空间名称和变量的名称
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v4=tf.get_variable("v",[1])
        print(v4.name)#输出foo/bar/v4:0.命名空间可以嵌套，同时以变量的名称也会加入所有命名空间的名称作为前缀。
    v5=tf.get_variable("v1",[1])
    print(v5.name)#输出foo/v1:0.当命名空间退出之后，变量名称也就不会在被加入其前缀了
#创建一个名称为空的命名空间，并设置为reuse=True
with tf.variable_scope("",reuse=True):
    v6=tf.get_variable("foo/bar/v",[1])#可以直接通过带命名空间名称的变量名来获取其他命名空间下的变量。比如这里通过指定名称foo/bar/v来获取在命名空间foo/bar/中创建的变量。输出True
    print(v6==v4)#True
    v7=tf.get_variable("foo/v1",[1])
    print(v7==v5)#True
#通过tf.variable_scope和tf.get_variable函数，以下代码对定义的计算前向传播结果的函数做了一些改进。




