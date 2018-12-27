import inspect
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf


# 去噪自编码器
class Dae_Engine(object):
    # 采用习惯用法定义常量
    TRAIN_MODE_NEW = 1
    TRAIN_MODE_CONTINUE = 2

    def __init__(self, name='dae', tf_graph=tf.Graph(),
                 n=784, hidden_size=1024):
        self.datasets_dir = 'datasets/'
        self.name = name
        self.random_seed = 1
        self.input_data = None  # 定义输入数据
        self.input_labels = None  # 定义输入信号标签
        self.keep_prob = None  # 定义当采用Dropout调整技术时，神经元保持活跃的比例
        self.layer_nodes = []
        self.train_step = None
        self.cost = None  # 代价函数
        # tensorflow objects
        self.tf_graph = tf_graph
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.loss_func = 'cross_entropy'
        self.enc_act_func = tf.nn.tanh
        self.dec_act_func = tf.nn.tanh
        self.num_epochs = 100
        self.batch_size = 256
        self.opt = 'adam'
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.corr_type = 'masking'
        self.corr_frac = 0.1
        self.regtype = 'l2'
        self.regcoef = 5e-4
        self.n = n
        self.hidden_size = hidden_size

    # def run(self):
    #     ckpt_file = 'work/{0}.ckpt'.format(self.name)
    #     img_file = 'datasets/test5.png'
    #     img = io.imread(img_file, as_grey=True)
    #     raw = [1 if x < 0.5 else 0 for x in img.reshape(784)]
    #     sample = np.array(raw)
    #     X_run = sample.reshape(1, 784)
    #     digit = -1
    #     with self.tf_graph.as_default():
    #         self.build_model()
    #         saver = tf.train.Saver()
    #         with tf.Session() as sess:
    #             sess.run(tf.global_variables_initializer())
    #             saver.restore(sess, ckpt_file)
    #             hidden_data, output_data = sess.run([self.a2, self.y_],
    #                                                 feed_dict={self.X: X_run})

    def train(self, X_train, X_validation, mode=TRAIN_MODE_NEW):
        ckpt_file = 'work/{0}.ckpt'.format(self.name)  # 定义模型的保存文件，每个去燥自动编码机的模型文件要不同，否则就会互相覆盖
        with self.tf_graph.as_default():  # 启动TensorFlow的graph
            self.build_model()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())  # 初始化全局变量
                for epoch in range(self.num_epochs):  # 对全量训练样本集训练
                    # X_train_prime = self.add_noise(sess, X_train,
                    #                                self.corr_frac)#这里需要修改，不需要人为加入噪声，这里默认数据集中有噪声
                    #shuff = list(zip(X_train, X_train_prime))  # 输入的内容需要修改的部分
                    shuff = list(X_train)
                    np.random.shuffle(shuff)
                    batches = [_ for _ in self.gen_mini_batches(shuff,
                                                                self.batch_size)]  # 按照指定大小拆分训练集大小,返回的是一个生成器对象
                    batch_idx = 1
                    for batch in batches:
                        X_batch_raw = batch
                        X_batch = np.array(X_batch_raw).astype(np.float32)
                        #X_prime_batch = np.array(X_prime_batch_raw). \
                         #   astype(np.float32)
                        batch_idx += 1
                        loss = sess.run([self.train_op, self.J],
                                             feed_dict={self.y: X_batch})  # 通过tf求出代价函数值
                        if batch_idx % 1000 == 0:
                            print('epoch{0}_batch{1}: {2}'.format(epoch,
                                                                  batch_idx, loss))
                            saver.save(sess, ckpt_file)

    # def add_noise(self, sess, X, corr_frac):  # 向样本集加入噪声
    #     X_prime = X.copy()
    #     rand = tf.random_uniform(X.shape)
    #     X_prime[sess.run(tf.nn.relu(tf.sign(corr_frac - rand))). \
    #         astype(np.bool)] = 0
    #     return X_prime

    def gen_mini_batches(self, X, batch_size):  # 产生迷你批次
        X = np.array(X)
        for i in range(0, X.shape[0], batch_size):
            yield X[i:i + batch_size]

    def build_model(self):
        print('Build Denoising Autoencoder Model')
        print('begine to build the model')
        self.X = tf.placeholder(shape=[None, self.n], dtype=tf.float32)  # 输入信号
        self.y = tf.placeholder(shape=[None, self.n], dtype=tf.float32)  # 输出信号
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')  # 采用Dropout调整项时，0神经元的活跃概率
        self.W1 = tf.Variable(
            tf.truncated_normal(
                shape=[self.n, self.hidden_size], mean=0.0, stddev=0.1),
            name='W1')  # 定义输入层到隐藏层的连接权值对象，采用均值为0，标准差为0.1的正态分布随机数进行初始化。
        self.b2 = tf.Variable(tf.constant(
            0.001, shape=[self.hidden_size]), name='b2')  # 隐藏层偏执，采用均值为0.标准差为0.01的正态分布随机数
        self.b3 = tf.Variable(tf.constant(
            0.001, shape=[self.n]), name='b3')  # 定义输出层偏执，均值0，标准差为0.01的正态分布随机数
        '''
        由于隐藏层到输出层的连接权值矩阵是输出层到隐藏层连接权值矩阵的转换，所以不需要定义隐藏层到输出层的连接权值矩阵
        '''
        '''定义输入层到隐藏层的编码机部分，首先将输入信号与连接权值矩阵相乘，再加上隐藏层偏执值，最后经过双曲正切激活函数，求出隐藏层
        输出。
        '''
        with tf.name_scope('encoder'):
            z2 = tf.matmul(self.X, self.W1) + self.b2
            self.a2 = tf.nn.tanh(z2)
        '''定义隐藏层到输出的解码机部分，将隐藏层输出信号与输入层到隐藏等连接权值矩阵W1的转置相乘，加上输出层的偏执，经过双曲正切激活函数，
        求出输出信号。
        '''
        with tf.name_scope('decoder'):
            z3 = tf.matmul(self.a2, tf.transpose(self.W1)) + self.b3
            a3 = tf.nn.tanh(z3)
        self.y_ = a3
        r_y_ = tf.clip_by_value(self.y_, 1e-10, float('inf'))
        r_1_y_ = tf.clip_by_value(1 - self.y_, 1e-10, float('inf'))  # 浮点数精度问题，若值小于1e-10,则赋值1e-10
        cost = - tf.reduce_mean(tf.add(
            tf.multiply(self.y, tf.log(r_y_)),
            tf.multiply(tf.subtract(1.0, self.y), tf.log(r_1_y_))))  # 定义代价函数
        self.J = cost + self.regcoef * tf.nn.l2_loss([self.W1])  # 定义最终的代价函数为加上L2调整项后的值。
        '''
        tf.nn.l2_loss函数用于计算权值衰减项，这里只对连接权值进行调整。self.regcoef 为连接权值衰减项的系数
        '''
        self.train_op = tf.train.AdamOptimizer(0.001, 0.9, 0.9, 1e-08). \
            minimize(self.J)  # 定义训练操作，采用Adam优化算法，求代价函数为最小值时的参数。
        #def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,use_locking=False, name="Adam"):

    def transform(self, graph, data):  #层与层之间互相做输入输出
        ckpt_file = 'work/{0}.ckpt'.format(self.name)  # 指定模型参数的保存文件，模型参数文件以自己的名称为前缀，因此不用编码器之间不会重复
        with self.tf_graph.as_default():
            saver = tf.train.Saver()  # 用于恢复模型的参数和超参数
            with tf.Session() as sess:
                saver.restore(sess, ckpt_file)  # 从模型参数文件中恢复模型的参数和超参数
                feed = {self.X: data, self.keep_prob: 1}
                return sess.run(self.a2, feed_dict=feed)
    '''
    当预训练完所有的去噪自动编码机之后，就进入了整体网络调优阶段，可以把这个阶段假想为在已经训练好的去噪自动编码机上叠加一个用于分类的softmax回归层，形成一个
    完整的多层感知器模型，相当于已经知道了多层感知器模型中的连接权值矩阵和偏执值，在此基础上训练多层感知器模型。
    '''