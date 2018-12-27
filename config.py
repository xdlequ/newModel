# 模型参数配置，单独列出文件，方便修改

class Config(object):
    def __init__(self):

        self.struct = [None, 1024, 512, 256, None]
        self.prev = None  # 不同的数据集输入带下不一样。需要后面赋值
        self.reg = 10
        self.lambd = 0.5  # lambda取值需要进行一些变化再测试。
        # 训练参数
        self.alpha = 1
        self.batch_size = 256
        self.num_sampled = 10
        self.max_iters = 20000
        self.learning_rate = 1e-4
