import inspect
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from StackAutoEncoder.app_global import FLAGS
from tensorflow.examples.tutorials.mnist import input_data
from StackAutoEncoder.mlp_engine import Mlp_Engine
from StackAutoEncoder.dae_engine import Dae_Engine

# 堆叠去噪自编码器
class Sda_Engine(object):
    # 采用习惯用法定义常量
    TRAIN_MODE_NEW = 1
    TRAIN_MODE_CONTINUE = 2

    def __init__(self):
        self.datasets_dir = 'datasets/'
        self.random_seed = 1  # 在网络权值和偏执值初始化时，都会用到随机数来进行初始化，如果每次生成的随机数不同，那么不便于调试。
        self.dae_W = []  # 连接权值矩阵列表
        self.dae_b = []  # 偏执值列表
        self.daes = []  # 逐层预训练的去噪自编码器列表
        self.dae_graphs = []  # 逐层预训练的去噪自编码器所对应的Graph列表。每个去噪自编码器对应一个Graph对象
        self.layers = [1024, 512, 256]
        self.name = 'sda'
        prev = 784
        self.mlp_engine = None  # 用于调优阶段的多层感知器模型
        for idx, layer in enumerate(self.layers):  # 循环初始化堆叠去噪自编码机的各个去噪自动编码器
            dae_str = 'dae_' + str(idx + 1)
            name = self.name + '_' + dae_str
            tf_graph = tf.Graph()  # 建立新的计算图
            self.daes.append(Dae_Engine(name, tf_graph=tf_graph, n=prev,
                                        hidden_size=layer))  # 生成新的去噪自动编码机，并添加到去噪自动编码机列表中
            prev = layer
            self.dae_graphs.append(tf_graph)  # 将生成的TensorFlow的Graph添加到Graph列表中。

    def run(self, ckpt_file='work/dae.ckpt'):
        if self.mlp_engine is None:
            self.mlp_engine = Mlp_Engine(self.daes, 'datasets')
        self.mlp_engine.run()

    def pretrain(self, X_train, X_test):
        X_train_prev = X_train
        X_test_prev = X_test  # 每个去燥自动编码机的输入均为上一个去燥自动编码机的隐藏层的内容，初始时将上一个去燥自动编码机隐藏层的值设置为训练样本集和测试集
        for idx, dae in enumerate(self.daes):  #
            print('pretrain:{0}'.format(dae.name))
            tf_graph = self.dae_graphs[idx]
            X_train_prev, X_test_prev = self.pretrain_dae(  # 预训练单个去噪自动编码器
                self.dae_graphs[idx], dae,
                X_train_prev, X_test_prev)
        return X_train_prev, X_test_prev

    def pretrain_dae(self, graph, dae, X_train, X_validation):  #
        dae.train(X_train, X_validation)  # 调用训练方法，，进行预训练
        X_train_next = dae.transform(graph, X_train)  # 调用去噪自动编码机的transform方法，将训练样本集转变为其隐藏层输出的结果，用于对下一个去噪自动编码机的输入
        X_validation_next = dae.transform(graph, X_validation)  # 同上，只不过训练集变了
        return X_train_next, X_validation_next

    def train(self, mode=TRAIN_MODE_NEW, ckpt_file='work/dae.ckpt'):
        X_train, y_train, X_validation, y_validation, \
        X_test, y_test, mnist = self.load_datasets()
        self.pretrain(X_train, X_validation)
        if self.mlp_engine is None:  # 初始化多层感知机模型
            self.mlp_engine = Mlp_Engine(self.daes, 'datasets')
        self.mlp_engine.train()  # 调用多层感知机模型的训练方法，对参数进行调优

    def build_model(self):
        print('Build stack denoising autoencoder')

    def load_datasets(self):
        mnist = input_data.read_data_sets(self.datasets_dir,
                                          one_hot=True)
        X_train = mnist.train.images
        y_train = mnist.train.labels
        X_validation = mnist.validation.images
        y_validation = mnist.validation.labels
        X_test = mnist.test.images
        y_test = mnist.test.labels
        return X_train, y_train, X_validation, y_validation, \
               X_test, y_test, mnist