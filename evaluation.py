import numpy as np
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import math
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity

#评估部分
def calculate_distance( embeddings, type): # N * emb_size
    if type == 'euclidean_distances':
        Y_predict = -1.0 * euclidean_distances(embeddings, embeddings)
        return Y_predict
    if type == 'cosine_similarity':
        Y_predict = cosine_similarity(embeddings, embeddings)
        return Y_predict

def norm(a):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * a[i]
    return math.sqrt(sum)

def cosine_similarity( a,  b):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * b[i]
    return sum/(norm(a) * norm(b))

def evaluate_ROC(X_test, Embeddings):
    y_true = [ X_test[i][2] for i in range(len(X_test))]
    y_predict = [ cosine_similarity(Embeddings[X_test[i][0],:], Embeddings[X_test[i][1], :]) for i in range(len(X_test))]
    roc = roc_auc_score(y_true, y_predict)
    if roc < 0.5:
        roc = 1 - roc
    return roc


def evaluate_MAP( node_neighbors_map, Embeddings, distance_measure):
    '''
    given the embeddings of nodes and the node_neighbors, return the MAP value 获取一种映射值
    :param node_neighbors_map: [node_id : neighbors_ids]
    :param nodes: a dictionary, ['node_id']--len(nodes) of id for nodes, one by one; ['node_attr']--a list of attrs for corresponding nodes
    :param Embeddings:  # nodes_number * (id_dim + attr_dim), row sequence is the same as nodes['node_id']
    :return: MAP value
    '''
    MAP = .0
    Y_true = np.zeros((len(node_neighbors_map), len(node_neighbors_map)))
    for node in node_neighbors_map:
        # prepare the y_true
        for neighbor in node_neighbors_map[node]:
            Y_true[node][neighbor] = 1

    print (distance_measure)
    Y_predict = calculate_distance(Embeddings,distance_measure)
    for node in node_neighbors_map:
        MAP +=  average_precision_score(Y_true[node,:], Y_predict[node,:])

    return MAP/len(node_neighbors_map)


def read_label(inputFileName):
    f = open(inputFileName, "r")
    lines = f.readlines()
    f.close()
    N = len(lines)
    y = np.zeros(N, dtype=int)
    for line in lines:
        l = line.strip("\n\r").split(" ")
        y[int(l[0])] = int(l[1])

    return y


def multiclass_node_classification_eval(X, y, ratio=0.2, rnd=2018):
    warnings.filterwarnings("ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=rnd)
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")

    return macro_f1, micro_f1


def link_prediction_ROC(inputFileName, Embeddings):
    f = open(inputFileName, "r")
    lines = f.readlines()
    f.close()

    X_test = []

    for line in lines:
        l = line.strip("\n\r").split(" ")
        X_test.append([int(l[0]), int(l[1]), int(l[2])])

    y_true = [X_test[i][2] for i in range(len(X_test))]
    y_predict = [cosine_similarity(Embeddings[X_test[i][0], :].reshape(
        1, -1), Embeddings[X_test[i][1], :].reshape(1, -1))[0, 0] for i in range(len(X_test))]
    auc = roc_auc_score(y_true, y_predict)

    if auc < 0.5:
        auc = 1 - auc

    return auc


def node_classification_F1(Embeddings, y):
    macro_f1_avg = 0
    micro_f1_avg = 0
    for i in range(10):
        rnd = np.random.randint(2018)
        macro_f1, micro_f1 = multiclass_node_classification_eval(
            Embeddings, y, 0.7, rnd)
        macro_f1_avg += macro_f1
        micro_f1_avg += micro_f1
    macro_f1_avg /= 10
    micro_f1_avg /= 10
    print("Macro_f1 average value: " + str(macro_f1_avg))
    print("Micro_f1 average value: " + str(micro_f1_avg))