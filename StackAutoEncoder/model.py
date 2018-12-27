import tensorflow as tf
from StackAutoEncoder.dae_engine import Dae_Engine


class Model(object):

    def __init__(self, config, N, dims, X_target, X_node):
        self.config = config
        self.N = N
        self.dims = dims
        # 标签和输入
        self.labels = tf.placeholder(tf.int32, shape=[None, 1])
        self.inputs = tf.placeholder(tf.int32, shape=[None])
        self.vertices_num = N  # 节点个数
        self.name = "sda"
        prev = N
        self.daes = []  # 逐层预训练的去噪自编码器列表
        self.dae_graphs = []  # 逐层预训练的去噪自编码器所对应的Graph列表。每个去噪自编码器对应一个Graph对象

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
        self.b_2 = {}
        struct = self.struct
        '''
                for idx, layer in enumerate(self.layers):  # 循环初始化堆叠去噪自编码机的各个去噪自动编码器
            dae_str = 'dae_' + str(idx + 1)
            name = self.name + '_' + dae_str
            tf_graph = tf.Graph()  # 建立新的计算图
            
            self.daes.append(Dae_Engine(name, tf_graph=tf_graph, n=prev,
                                        hidden_size=layer))  # 生成新的去噪自动编码机，并添加到去噪自动编码机列表中
            prev = layer  # 不断更新输入层大小
            self.dae_graphs.append(tf_graph)  # 将生成的TensorFlow的Graph添加到Graph列表中。
        '''
        # # 定义多组编码器 方案1
        # for idx, layer in enumerate(self.struct):
        #     dae_str = 'dae_' + str(idx+1)
        #     name = self.name + '_'+dae_str
        #     tf_graph = tf.Graph()
        #     self.daes.append(Dae_Engine(name, tf_graph=tf_graph, n=prev,
        #                                 hidden_size=layer))  # 每一个去噪自编码器内部集成了编码器和解码器
        #     prev = layer
        #     self.dae_graphs.append(tf_graph)
        for i in range(0, self.layers - 1):
            dae_str = 'dae_' + str(i + 1)
            name_W = dae_str + 'encoder' + '_W'
            name_b = dae_str + 'encoder' + '_b'
            # 定义输入层到隐藏层的连接权值对象，采用均值为0，标准差为0.1的正态分布随机数进行初始化。
            self.W[name_W] = tf.Variable(tf.truncated_normal(
                shape=[struct[i], struct[i+1]], mean=0.0, stddev=0.1), name=name_W)
            self.b[name_b] = tf.Variable(tf.constant(
                0.001, shape=[struct[i+1]]), name=name_b)  # 隐藏层偏执，采用均值为0.标准差为0.01的正态分布随机数
        struct.reverse()
        for i in range(0, self.layers - 1):
            dae_str = 'dae_' + str(i + 1)
            name_W = dae_str + 'decoder' + '_W'
            name_b_2 = dae_str + 'decoder' + '_b_2'
            self.W[name_W] = tf.get_variable(
                name_W, [struct[i], struct[i + 1]], initializer=tf.contrib.layers.xavier_initializer())
            self.b[name_b_2] = tf.get_variable(
                name_b_2, [struct[i + 1]], initializer=tf.zeros_initializer())
        self.struct.reverse()
        '''
        tf.contrib.layers.xavier_initializer()  # 一种带权重的初始化方法
        该初始化器旨在使所有层中的梯度比例保持大致相同。
        在均匀分布中，这最终是范围： x = sqrt(6. / (in + out)); [-x, x]并且对于正态分布，
        使用标准偏差sqrt(2. / (in + out))。
        '''

        # decode module 解码器部分
        # struct.reverse()
        # for i in range(0, self.layers - 1):
        #     name_W = "decoder_W_" + str(i)
        #     name_b = "decoder_b_" + str(i)
        #     self.W[name_W] = tf.get_variable(
        #         name_W, [struct[i], struct[i + 1]], initializer=tf.contrib.layers.xavier_initializer())
        #     self.b[name_b] = tf.get_variable(
        #         name_b, [struct[i + 1]], initializer=tf.zeros_initializer())
        # self.struct.reverse()
        self.X = tf.placeholder(tf.float32, shape=[None, config.struct[0]])  # X特质特征
        # define input
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
        # self.X = tf.placeholder(tf.float32, shape=[None, self.config.struct[0]])  # X特质特征
        # self.y = tf.placeholder(tf.float32, [None, self.dims])  # 定义正确的标签结果
        # with self.daes[0].tf_graph.as_default():
        #     with tf.Session() as sess:
        #         sess.run(tf.global_variables_initializer())
        #         dae0_W1 = sess.run(self.daes[0].W1)
        #         dae0_b2 = sess.run(self.daes[0].b2)
        #         dae0_y_b = sess.run(self.daes[0].b3)
        # with self.daes[1].tf_graph.as_default():
        #     with tf.Session() as sess:
        #         sess.run(tf.global_variables_initializer())
        #         dae1_W1 = sess.run(self.daes[1].W1)
        #         dae1_b2 = sess.run(self.daes[1].b2)
        #         dae1_y_b = sess.run(self.daes[1].b3)
        # with self.daes[2].tf_graph.as_default():
        #     with tf.Session() as sess:
        #         sess.run(tf.global_variables_initializer())
        #         dae2_W1 = sess.run(self.daes[2].W1)
        #         dae2_b2 = sess.run(self.daes[2].b2)
        #         dae2_y_b = sess.run(self.daes[2].b3)
        # with self.daes[3].tf_graph.as_default():
        #     with tf.Session() as sess:
        #         sess.run(tf.global_variables_initializer())
        #         dae3_W1 = sess.run(self.daes[3].W1)
        #         dae3_b2 = sess.run(self.daes[3].b2)
        #         dae3_y_b = sess.run(self.daes[3].b3)
        # print("encoder")
        # # 第一个去噪自动编码器 输入到1024
        # W_1 = tf.Variable(dae0_W1, name='W_1')
        # b_2 = tf.Variable(dae0_b2, name='b_2')
        # z_2 = tf.matmul(self.X, W_1) + b_2  # 第二层的输入
        # a_2 = tf.nn.tanh(z_2)  # tf.nn.relu(self.z_2) 利用双曲正切函数求出第二层的原始输出。
        # # 从1024到512的去噪自动编码机
        # W_2 = tf.Variable(dae1_W1, name='W_2')
        # b_3 = tf.Variable(dae1_b2, name='b_3')
        # z_3 = tf.matmul(a_2, W_2) + b_3
        # a_3 = tf.nn.tanh(z_3)  # tf.nn.relu(self.z_3)
        # # 从512到256的去噪自动编码机
        # W_3 = tf.Variable(dae2_W1, name='W_3')
        # b_4 = tf.Variable(dae2_b2, name='b_4')
        # z_4 = tf.matmul(a_3, W_3) + b_4
        # a_4 = tf.nn.tanh(z_4)  # tf.nn.relu(self.z_4)
        # # 从256到128的输出层
        #
        # W_4 = tf.Variable(dae3_W1, name='W_4')
        # b_5 = tf.Variable(dae3_b2, name='b_5')
        # z_5 = tf.matmul(a_4, W_4) + b_5
        # self.embedding_result = z_5  # 利用softmax函数求出第6层的输出。即计算出的标签分类结果
        # print("decoder")
        # # 反向第一层
        # y_b_5 = tf.Variable(tf.zeros([256]))
        # y_z_5 = tf.matmul(z_5, tf.transpose(W_4)) + y_b_5  # 解码器的输出
        # #
        # y_b_4 = tf.Variable(dae2_y_b, name='y_b_3')
        # y_z_4 = tf.matmul(z_4, tf.transpose(W_3)) + y_b_4  # 解码器的输出
        # #
        # y_b_3 = tf.Variable(dae1_y_b, name='y_b_3')
        # y_z_3 = tf.matmul(z_3, tf.transpose(W_2)) + y_b_3  # 解码器的输出
        # #
        # y_b_2 = tf.Variable(dae0_y_b, name='y_b_2')
        # y_z_2 = tf.matmul(z_2, tf.transpose(W_1)) + y_b_2  # 解码器的输出

        def encoder(X):
            for i in range(0, self.layers - 1):
                dae_str = 'dae_' + str(i + 1)
                name_W = dae_str + 'encoder' + '_W'
                name_b = dae_str + 'encoder' + '_b'
                X = tf.nn.tanh(tf.matmul(X, self.W[name_W]) + self.b[name_b])
            return X

        def decoder(X):
            for i in range(0, self.layers - 1):
                dae_str = 'dae_' + str(i + 1)
                name_W = dae_str + 'decoder' + '_W'
                name_b_2 = dae_str + 'decoder' + '_b_2'
                if name_b_2 in self.b:
                    print('exist++++++')
                    print(self.b[name_b_2])
                else:
                    print('not exist')
                X = tf.nn.tanh(tf.matmul(X, self.W[name_W]) + self.b[name_b_2])
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

