import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, config, N, dims, X_target, X_node):
        self.config = config
        self.N = N
        self.dims = dims
        # 标签和输入
        self.labels = tf.placeholder(tf.int32, shape=[None, 1])
        self.inputs = tf.placeholder(tf.int32, shape=[None])
        self.vertices_num = N  # 节点个数

        # 生成一个给定值的常量张量
        self.X_target = tf.constant(X_target, dtype=tf.float32)
        self.X_node = tf.constant(X_node, dtype=tf.float32)

        '''
        tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
        tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引，
        '''
        alpha = tf.get_variable("alpha", shape=[self.vertices_num, 1], initializer=tf.zeros_initializer())
        input_wgt = tf.sigmoid(tf.nn.embedding_lookup(alpha, self.inputs))
        self.X_new = tf.nn.embedding_lookup(self.X_target, self.inputs)  # 邻居节点的属性均值
        self.X_ori = tf.nn.embedding_lookup(self.X_node, self.inputs)  # 源节点的属性值

        #self.X_input = config.lambd * (self.X_ori) + (1 - config.lambd) * (self.X_new)
        self.X_input = tf.multiply(input_wgt, self.X_ori) + tf.multiply((1-input_wgt), self.X_new)

        # 为编码器定义参数
        self.layers = len(config.struct)
        self.struct = config.struct
        self.W = {}
        self.b = {}
        struct = self.struct
        print(struct)
        # encode module 编码器部分
        for i in range(0, self.layers - 1):
            name_W = "encoder_W_" + str(i)
            name_b = "encoder_b_" + str(i)
            # print(struct[i],struct[i+1])
            self.W[name_W] = tf.get_variable(
                name_W, [struct[i], struct[i + 1]], initializer=tf.contrib.layers.xavier_initializer())  # 一种带权重的初始化方法
            '''
            tf.contrib.layers.xavier_initializer()  # 一种带权重的初始化方法
            该初始化器旨在使所有层中的梯度比例保持大致相同。
            在均匀分布中，这最终是范围： x = sqrt(6. / (in + out)); [-x, x]并且对于正态分布，
            使用标准偏差sqrt(2. / (in + out))。
            '''
            self.b[name_b] = tf.get_variable(
                name_b, [struct[i + 1]], initializer=tf.zeros_initializer())  # 张量初始化为0

        # decode module 解码器部分
        struct.reverse()
        for i in range(0, self.layers - 1):
            name_W = "decoder_W_" + str(i)
            name_b = "decoder_b_" + str(i)
            self.W[name_W] = tf.get_variable(
                name_W, [struct[i], struct[i + 1]], initializer=tf.contrib.layers.xavier_initializer())
            self.b[name_b] = tf.get_variable(
                name_b, [struct[i + 1]], initializer=tf.zeros_initializer())
        self.struct.reverse()

        # define input
        self.X = tf.placeholder(tf.float32, shape=[None, config.struct[0]])  # X特质特征

        self.make_compute_graph()  # 前馈传播，训练
        self.make_autoencoder_loss()  # 计算损失函数

        # compute gradients for deep autoencoder 通过梯度下降法对loss_ae进行优化. self.loss_ae是总的损失函数
        self.train_opt_ae = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss_ae)

        # define variables for skipgram
        # construct variables for nce loss
        self.nce_weights = tf.get_variable("nce_weights",
                                           [self.N, self.dims], initializer=tf.contrib.layers.xavier_initializer())
        self.nce_biases = tf.get_variable(
            "nce_biases", [self.N], initializer=tf.zeros_initializer())
        self.loss_sg = self.make_skipgram_loss()

        # compute gradients for skipgram
        self.train_opt_sg = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss_sg)

    '''
    tf.nn.sampled_softmax_loss
    计算并返回采样的softmax训练损失。

    这是在大量课程中训练softmax分类器的更快方法。

    此操作仅用于培训。通常低估了完全的softmax损失。

    常见的用例是使用此方法进行训练，并计算完整的softmax损失以进行评估或推断。
    '''

    def make_skipgram_loss(self):
        loss = tf.reduce_sum(tf.nn.sampled_softmax_loss(
            weights=self.nce_weights,
            biases=self.nce_biases,
            labels=self.labels,  # 标签用于分类任务
            inputs=self.Y,  # Y对应隐层获取的特征大小为128
            num_sampled=self.config.num_sampled,  # 负采样大小
            num_classes=self.N  # N对应图中节点的个数
        ))
        return loss

    def make_compute_graph(self):
        def encoder(X):
            for i in range(0, self.layers - 1):
                name_W = "encoder_W_" + str(i)
                name_b = "encoder_b_" + str(i)
                X = tf.nn.tanh(tf.matmul(X, self.W[name_W]) + self.b[name_b])
            return X

        def decoder(X):
            for i in range(0, self.layers - 1):
                name_W = "decoder_W_" + str(i)
                name_b = "decoder_b_" + str(i)
                X = tf.nn.tanh(tf.matmul(X, self.W[name_W]) + self.b[name_b])
            return X

        self.Y = encoder(self.X)
        self.X_reconstruct = decoder(self.Y)

    def make_autoencoder_loss(self):

        def get_autoencoder_loss(X, newX):
            return tf.reduce_sum(tf.pow((newX - X), 2))

        def get_reg_loss(weights, biases):
            # l2_loss 这个函数的作用是利用 L2 范数来计算张量的误差值，但是没有开方并且只取 L2 范数的值的一半
            reg = tf.add_n([tf.nn.l2_loss(w) for w in weights.values()])
            reg += tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])
            return reg

        loss_autoencoder = get_autoencoder_loss(self.X_new, self.X_reconstruct)
        loss_reg = get_reg_loss(self.W, self.b)
        self.loss_ae = self.config.alpha * loss_autoencoder + self.config.reg * loss_reg
