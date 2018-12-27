from qutils import *
import tensorflow as tf
from StackAutoEncoder.dae_engine import Dae_Engine


def pretrain_dae(graph, dae, X_train):  #
    dae.train(X_train)  # 调用训练方法，，进行预训练
    X_train_next = dae.transform(graph, X_train)  # 调用去噪自动编码机的transform方法，将训练样本集转变为其隐藏层输出的结果，用于对下一个去噪自动编码机的输入
    return X_train_next


# 训练过程分为逐层预训练和整体调优
class Newmodel(object):
    def __init__(self, config, N, dims, X_target, X_node):
        self.random_seed = 2018  # 在网络权值和偏执值初始化时，都会用到随机数来进行初始化，如果每次生成的随机数不同，那么不便于调试。
        self.dae_W = []  # 连接权值矩阵列表
        self.dae_b = []  # 偏执值列表
        self.daes = []  # 逐层预训练的去噪自编码器列表
        self.dae_graphs = []  # 逐层预训练的去噪自编码器所对应的Graph列表。每个去噪自编码器对应一个Graph对象
        self.prev = N  # 输入层大小为初始时节点的个数。
        self.name = "sda"
        self.config = config
        self.layers = config.struct
        self.layers[0] = N
        self.layers[-1] = dims
        self.k = dims
        self.mlp_engine = None  # 用于调优阶段的多层感知器模型
        self.inputs = tf.placeholder(tf.int32, shape=[None])
        self.X_target = tf.constant(X_target, dtype=tf.float32)
        self.X_node = tf.constant(X_node, dtype=tf.float32)
        alpha = tf.get_variable("alpha", shape=[self.prev, 1], initializer=tf.zeros_initializer())
        input_wgt = tf.sigmoid(tf.nn.embedding_lookup(alpha, self.inputs))
        self.X_new = tf.nn.embedding_lookup(self.X_target, self.inputs)  # 邻居节点的属性均值
        self.X_ori = tf.nn.embedding_lookup(self.X_node, self.inputs)  # 源节点的属性值
        self.X_input = tf.multiply(input_wgt, self.X_ori) + tf.multiply((1 - input_wgt), self.X_new)
        self.X = tf.placeholder(tf.float32, shape=[None, N])
        # --------------------------------------------------------

        self.batch_size = config.batch_size  # 定义输出信号大小为10，即分了10个类
        self.lanmeda = 0.001  # L2正则化的系数
        self.keep_prob_val = 0.75  # 采用Dropout调整技术时，神经元激活的概率。这个不一定会用
        self.labels = tf.placeholder(tf.int32, shape=[None, 1])
        self.model = {}
        self.W = {}
        self.b = {}
        # --------------------------------------------------------
        self.loss_sg = self.make_skipgram_loss()
        self.train_opt_sg = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss_sg)

        self.train_step_ae = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9,
                                                    beta2=0.999, epsilon=1e-08, use_locking=False,
                                                    name='Adam').minimize(self.loss_ae)
        for idx, layer in enumerate(self.layers):  # 循环初始化堆叠去噪自编码机的各个去噪自动编码器
            dae_str = 'dae_' + str(idx + 1)
            name = self.name + '_' + dae_str
            tf_graph = tf.Graph()  # 建立新的计算图
            prev = layer  # 不断更新输入层大小
            self.daes.append(Dae_Engine(name, tf_graph=tf_graph, n=prev,
                                        hidden_size=layer))  # 生成新的去噪自动编码机，并添加到去噪自动编码机列表中

            self.dae_graphs.append(tf_graph)  # 将生成的TensorFlow的Graph添加到Graph列表中。
        # X_node X_target，对应的是输入部分。

    def pretrain(self, X_train):  # 逐层预训练
        X_train_prev = X_train
        for idx, dae in enumerate(self.daes):  #
            print('pretrain:{0}'.format(dae.name))
            tf_graph = self.dae_graphs[idx]
            X_train_prev = pretrain_dae(  # 预训练单个去噪自动编码器
                self.dae_graphs[idx], dae,
                X_train_prev)
        return X_train_prev

    def train(self, X_train):
        self.pretrain(X_train)  # 预训练
        if self.mlp_engine is None:  # 初始化多层感知机模型
            self.mlp_engine = self.bulid_model(self.X_input)

    def build_model(self, input, mode='train'):
        self.X = input  # 输入信号
        self.y = tf.placeholder(tf.float32, [None, self.k])  # 定义正确的标签结果
        # self.keep_prob = tf.placeholder(tf.float32)  # Dropout失活率
        if 'train' == mode:
            # 取出预训练去噪自动编码机参数
            with self.daes[0].tf_graph.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    dae0_W1 = sess.run(self.daes[0].W1)
                    dae0_b2 = sess.run(self.daes[0].b2)
                    dae0_y_b = sess.run(self.daes[0].b3)
            with self.daes[1].tf_graph.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    dae1_W1 = sess.run(self.daes[1].W1)
                    dae1_b2 = sess.run(self.daes[1].b2)
                    dae1_y_b = sess.run(self.daes[1].b3)
            with self.daes[2].tf_graph.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    dae2_W1 = sess.run(self.daes[2].W1)
                    dae2_b2 = sess.run(self.daes[2].b2)
                    dae2_y_b = sess.run(self.daes[2].b3)
            # with self.daes[3].tf_graph.as_default():
            #     with tf.Session() as sess:
            #         sess.run(tf.global_variables_initializer())
            #         dae3_W1 = sess.run(self.daes[3].W1)
            #         dae3_b2 = sess.run(self.daes[3].b2)
        print("encoder model and decoder model")
        # 第一个去噪自动编码器 输入到1024
        if 'train' == mode:  # 创建训练模型
            self.W_1 = tf.Variable(dae0_W1, name='W_1')
            self.b_2 = tf.Variable(dae0_b2, name='b_2')
            self.y_b_2 = tf.Variable(dae0_y_b, name='y_b_2')
        else:  # 创建运行状态的模型
            self.W_1 = tf.Variable(tf.truncated_normal([self.prev, 1024], mean=0.0,
                                                       stddev=0.1), name='W_1')
            self.b_2 = tf.Variable(tf.zeros([1024]), name='b_2')
            self.y_b_2 = tf.Variable(tf.zeros[self.prev], name='y_b_2')
        self.z_2 = tf.matmul(self.X, self.W_1) + self.b_2  # 第二层的输入
        self.y_z_2 = tf.matmul(self.z_2, tf.transpose(self.W_1)) + self.y_b_2  # 解码器的输出
        self.a_2 = tf.nn.tanh(self.z_2)  # tf.nn.relu(self.z_2) 利用双曲正切函数求出第二层的原始输出。
        self.W[0] = self.W_1
        self.b[0] = self.b_2
        # self.a_2_dropout = tf.nn.dropout(self.a_2, self.keep_prob)  # 使用dropout方法
        # 从1024到512的去噪自动编码机
        if 'train' == mode:
            self.W_2 = tf.Variable(dae1_W1, name='W_2')
            self.b_3 = tf.Variable(dae1_b2, name='b_3')
            self.y_b_3 = tf.Variable(dae1_y_b, name='y_b_3')
        else:
            self.W_2 = tf.Variable(tf.truncated_normal([1024, 512], mean=0.0,
                                                       stddev=0.1), name='W_2')
            self.b_3 = tf.Variable(tf.zeros([512]), name='b_3')
            self.y_b_3 = tf.Variable(tf.zeros[1024], name='y_b_3')
        self.z_3 = tf.matmul(self.a_2, self.W_2) + self.b_3
        self.y_z_3 = tf.matmul(self.z_3, tf.transpose(self.W_2)) + self.y_b_3  # 解码器的输出
        self.a_3 = tf.nn.tanh(self.z_3)  # tf.nn.relu(self.z_3)
        self.W[1] = self.W_2
        self.b[1] = self.b_3
        # self.a_3_dropout = tf.nn.dropout(self.a_3, self.keep_prob)
        # 从512到256的去噪自动编码器
        if 'train' == mode:
            self.W_3 = tf.Variable(dae2_W1, name='W_3')
            self.b_4 = tf.Variable(dae2_b2, name='b_4')
            self.y_b_4 = tf.Variable(dae2_y_b, name='y_b_3')
        else:
            self.W_3 = tf.Variable(tf.truncated_normal([512, 256], mean=0.0,
                                                       stddev=0.1), name='W_3')
            self.b_4 = tf.Variable(tf.zeros([256]), name='b_4')
            self.y_b_4 = tf.Variable(tf.zeros[512], name='y_b_4')
        self.z_4 = tf.matmul(self.a_3, self.W_3) + self.b_4

        self.y_z_4 = tf.matmul(self.z_4, tf.transpose(self.W_3)) + self.y_b_4  # 解码器的输出
        self.a_4 = tf.nn.tanh(self.z_4)  # tf.nn.relu(self.z_4)
        self.W[2] = self.W_3
        self.b[2] = self.b_4
        # self.a_4_dropout = tf.nn.dropout(self.a_4, self.keep_prob)
        # 输出层 256到输出
        self.W_4 = tf.Variable(tf.zeros([256, self.k]))
        self.b_5 = tf.Variable(tf.zeros([self.k]))
        self.y_b_5 = tf.Variable(tf.zeros([256]))
        self.z_5 = tf.matmul(self.a_4, self.W_4) + self.b_5
        self.y_z_5 = tf.matmul(self.z_5, tf.transpose(self.W_4)) + self.y_b_5  # 解码器的输出
        self.W[3] = self.W_4
        self.b[3] = self.b_5

        # 训练部分
        # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(
        #     self.y,reduction_indices=[1]))   #交叉熵的计算。
        # train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
        self.compute_loss_ae()
        # 采用梯度下降法优化，求使代价函数达到最小值时连接权值W和偏移量b的值

        # compute gradients for skipgram
        return self.X, self.y

    def compute_loss_ae(self):
        def get_autoencoder_loss(X, newX):
            return tf.reduce_sum(tf.pow((newX - X), 2))

        def get_reg_loss(weights, biases):
            # l2_loss 这个函数的作用是利用 L2 范数来计算张量的误差值，但是没有开方并且只取 L2 范数的值的一半
            reg = tf.add_n([tf.nn.l2_loss(w) for w in weights.values()])
            reg += tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])
            return reg

        loss_autoencoder = get_autoencoder_loss(self.y_z_5, self.z_4) + get_autoencoder_loss(self.y_z_4, self.z_3) \
                           + get_autoencoder_loss(self.y_z_3, self.z_2) + get_autoencoder_loss(self.y_z_2, self.X)
        loss_reg = get_reg_loss(self.W, self.b)
        self.loss_ae = self.config.alpha * loss_autoencoder + self.config.reg * loss_reg

    def make_skipgram_loss(self):
        self.nce_weights = tf.get_variable("nce_weights",
                                           [self.prev, self.k], initializer=tf.contrib.layers.xavier_initializer())
        self.nce_biases = tf.get_variable(
            "nce_biases", [self.prev], initializer=tf.zeros_initializer())
        loss_sg = tf.reduce_sum(tf.nn.sampled_softmax_loss(
            weights=self.nce_weights,
            biases=self.nce_biases,
            labels=self.labels,  # 标签用于分类任务
            inputs=self.k,  # Y对应隐层获取的特征大小为128
            num_sampled=self.config.num_sampled,  # 负采样大小
            num_classes=tf.cast(self.prev, tf.float32)  # N对应图中节点的个数
        ))
        return loss_sg

    '''
    daes,input,config,layers,datasets_dir
    '''
