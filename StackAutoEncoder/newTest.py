import node2vec
import qutils
import time
from evaluation import *
from config import Config
import tensorflow as tf
from StackAutoEncoder.newmodel import Newmodel

# 测试node2vec方法
tf.app.flags.DEFINE_string("datasets", "citeseer", "datasets descriptions")
tf.app.flags.DEFINE_string(
    "inputEdgeFile", "../data/citeseer.edgelist", "input graph edge file")
tf.app.flags.DEFINE_string(
    "inputFeatureFile", "../data/citeseer.feature", "input graph feature file")
tf.app.flags.DEFINE_string(
    "inputLabelFile", "../data/citeseer.label", "input graph label file")
tf.app.flags.DEFINE_string(
    "outputEmbedFile", "../data/citeseer.embed", "output embedding result")
tf.app.flags.DEFINE_integer("dimensions", 128, "embedding dimensions")  # 定义输入大小为256
tf.app.flags.DEFINE_integer("feaDims", 3703, "feature dimensions")  # 目前来看attribute对应的是邻接矩阵
tf.app.flags.DEFINE_integer("walk_length", 80, "walk length")  # node2vec的一些设置节点数80，每次游走10步，窗口大小10
tf.app.flags.DEFINE_integer("num_walks", 10, "number of walks")  # 对应参数为γ
tf.app.flags.DEFINE_integer("window_size", 10, "window size")
tf.app.flags.DEFINE_float("p", 1.0, "p value")  # node2vec中的p和q
tf.app.flags.DEFINE_float("q", 1.0, "q value")
tf.app.flags.DEFINE_boolean("weighted", False, "weighted edges")  # 无权重的图
tf.app.flags.DEFINE_boolean("directed", False, "undirected edges")  # 无向图

def construct_neighbor(nx_G, X, FLAGS, mode="WAN"):
    '''
    为对应的节点创建好相应的邻居节点
    construct target neighbor feature matrix
    :param nx_G: 根据给定数据集创建好的图
    :param X: 读取的特征
    :param FLAGS: 初始化时设定的参数
    :param mode: 构造节点邻居的模式
    :return:
    construct target neighbor feature matrix
    '''

    X_target = np.zeros(X.shape)
    X_node = np.zeros(X.shape)
    nodes = nx_G.nodes()
    if mode == "OWN":  # autoencoder for reconstructing itself
        return X
    elif mode == "WAN":  # autoencoder for reconstructing Weighted Average Neighbor
        for node in nodes:
            neighbors = nx_G.neighbors(node)
            X_node[node] = X[node]
            if len(list(neighbors)) == 0:
                X_target[node] = X[node]  # 若node无邻接点，则将该节点的表示赋值给X_target
            else:
                temp = X[node]
                for n in neighbors:
                    #weight = cos_sim(X_node[node], X[n])
                    if FLAGS.weighted:  # 若是有权图
                        # weighted sum
                        pass
                    else:
                        #temp = np.vstack((temp, np.array(X[n])*weight))  # 将temp与X[n]垂直拼接
                        temp = np.vstack((temp,X[n]))
                temp = np.mean(temp, axis=0)  # 计算好了取均值
                X_target[node] = temp  # X_target代表节点邻居属性的均值
        return X_node, X_target


def generate_graph_context_all_pairs(path, window_size):
    '''
    generate graph context pairs
    :param path: 之前得到的游走序列。
    :param window_size:游走序列的窗口大小
    :return:
    游走产生的路径是一个二维矩阵的形式。
    '''
    all_pairs = []
    for k in range(len(path)):
        for i in range(len(path[k])):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 0 or j >= len(path[k]):
                    continue
                else:
                    all_pairs.append([path[k][i], path[k][j]])
    return np.array(all_pairs, dtype=np.int32)


def graph_context_batch_iter(all_pairs, batch_size):
    while True:
        start_idx = np.random.randint(0, len(all_pairs) - batch_size)  # 随机产生一个值作为起点
        batch_idx = np.array(range(start_idx, start_idx + batch_size))  # 生成一个batch_index序列
        batch_idx = np.random.permutation(batch_idx)  # 打乱序列
        batch = np.zeros(batch_size, dtype=np.int32)  # 行
        features = np.zeros((batch_size, 1), dtype=np.int32)  # 列
        batch[:] = all_pairs[batch_idx, 0]
        features[:, 0] = all_pairs[batch_idx, 1]
        yield batch, features


def main():
    # 使用node2vec先 生成必要的节点序列
    FLAGS = tf.app.flags.FLAGS
    inputEdgeFile = FLAGS.inputEdgeFile
    inputLabelFile = FLAGS.inputLabelFile
    inputFeatureFile = FLAGS.inputFeatureFile
    outputEmbedFile = FLAGS.outputEmbedFile
    nx_G = qutils.read_graph(FLAGS, inputEdgeFile)
    G = node2vec.Graph(nx_G, FLAGS.directed, FLAGS.p, FLAGS.q)
    window_size = FLAGS.window_size
    G.preprocess_transition_probs()
    walks = G.simulate_walks(FLAGS.num_walks, FLAGS.walk_length)  # node2vec的随机游走产生
    # Read features
    print("reading features...")
    X = qutils.read_feature(inputFeatureFile)  # 获取特征，这里的特征对应文章中的attribute，邻接矩阵

    print("generating graph context pairs ...")
    start_time = time.time()
    all_pairs = generate_graph_context_all_pairs(walks, window_size)
    end_time = time.time()
    print("time consumed for constructing graph context: %.2f" % (end_time - start_time))

    print("get the target neighbors for Graph...")
    nodes = nx_G.nodes()
    X_node, X_target = construct_neighbor(nx_G, X, FLAGS, mode="WAN")
    # X_node 对应的是节点的属性，X_target对应的是节点邻接点属性的均值
    N = len(nodes)
    feaDims = FLAGS.feaDims
    dims = FLAGS.dimensions

    config = Config()
    config.struct[0] = feaDims
    config.struct[-1] = dims
    model = Newmodel(config, N, dims, X_target, X_node)

    '''初始化tf的配置信息'''
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    batch_size = config.batch_size
    max_iters = config.max_iters
    embedding_result = None

    idx = 0
    print_every_k_iterations = 1000  # 每迭代1000次，打印一下
    start = time.time()

    total_loss = 0
    loss_sg = 0
    loss_ae = 0
    # for i in range(10):
    #     batch_index, batch_labels = next(
    #         graph_context_batch_iter(all_pairs, batch_size)
    #     )
    #     print("++++++++++++=")
    #     print(batch_index,batch_labels)
    #     print("--------------")
    for iter_cn in range(max_iters):
        idx += 1

        batch_index, batch_labels = next(
            graph_context_batch_iter(all_pairs, batch_size)
        )

        # train for autoencoder model
        start_idx = np.random.randint(0, N - batch_size)  # 开始下标
        batch_idx = np.array(range(start_idx, start_idx + batch_size))  # 一个batch_size大小的列表。
        batch_idx = np.random.permutation(batch_idx)
        batch_X = X[batch_idx]  # batch对应节点的属性信息。
        '''
        feed_dict是一个字典，在字典中需要给出每一个用到的占位符的取值。
        在训练神经网络时需要每次提供一个批量的训练样本，如果每次迭代选取的数据要通过常量表示，
        那么TensorFlow 的计算图会非常大。因为每增加一个常量，TensorFlow 都会在计算图中增加一个结点。
        所以说拥有几百万次迭代的神经网络会拥有极其庞大的计算图，而占位符却可以解决这一点，
        它只会拥有占位符这一个结点
        '''
        feed_dict = {model.X: batch_X, model.inputs: batch_idx}  # 建立一个一一映射的字典
        _, loss_ae_value = sess.run([model.train_step_ae, model.loss_ae], feed_dict=feed_dict)
        loss_ae += loss_ae_value

        # train for skip-gram model
        batch_X = X[batch_index]
        feed_dict = {model.X: batch_X, model.labels: batch_labels}
        _, loss_sg_value = sess.run(
            [model.train_opt_sg, model.loss_sg], feed_dict=feed_dict)
        loss_sg += loss_sg_value

        if idx % print_every_k_iterations == 0:
            end = time.time()

            print("iterations: %d" % (idx) + ", time elapsed: %.2f," % (end - start))
            total_loss = loss_sg / idx + loss_ae / idx
            print("loss: %.2f," % (total_loss))

            y = read_label(inputLabelFile)
            embedding_result = sess.run(model.Y, feed_dict={model.X: X})
            macro_f1, micro_f1 = multiclass_node_classification_eval(embedding_result, y, 0.7)
            print("[macro_f1 = %.4f, micro_f1 = %.4f]" % (macro_f1, micro_f1))

    print("optimization finished...")
    y = read_label(inputLabelFile)
    embedding_result = sess.run(model.Y, feed_dict={model.X: X})
    print("repeat 10 times for node classification with random split...")
    node_classification_F1(embedding_result, y)
    print("saving embedding result...")
    qutils.write_embedding(embedding_result, outputEmbedFile)


if __name__ == "__main__":
    main()
