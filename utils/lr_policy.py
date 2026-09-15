import math
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

class BaseLR():
    __metaclass__ = ABCMeta
    @abstractmethod
    def get_lr(self, cur_iter): pass


class WarmUp_LR(BaseLR):
    def __init__(self, base_lr, total_steps, warmup_ratio,
                 Types="linear", min_lr=1.0e-7):
        """
        Types: linear, cos，stepLR
        """
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.total_step = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        if Types == "linear":
            self.lr = self.Linear_lr()
        if Types == "cos":
            self.lr = self.Cos_lr()
        if Types == "ploy":
            self.lr = self.Ploy_lr()


    def get_lr(self, cur_iter):
        return self.lr[cur_iter]

    def Linear_lr(self):
        lr = []
        k1 = (self.base_lr - 1.0e-8) / (self.warmup_steps - 1)
        b1 = self.base_lr - k1 * self.warmup_steps

        k2 = -(self.base_lr - self.min_lr) / (self.total_step - self.warmup_steps)
        b2 = self.base_lr - k2 * self.warmup_steps

        for i in range(self.total_step):
            if i <= self.warmup_steps:
                lr.append(k1 * i + b1)
            else:
                lr.append(k2 * i + b2)
        return lr

    def Cos_lr(self):
        """
        - `T_max` 是一个周期内的迭代次数。在这个周期结束时，学习率会下降到 `eta_min`。
        - `eta_min` 是学习率下降的最小值。
        - `last_epoch` 记录当前迭代次数。
        """
        T_max = self.total_step - self.warmup_steps
        eta_min = self.min_lr
        lr = []
        k1 = (self.base_lr - 1.0e-8) / (self.warmup_steps - 1)
        b1 = self.base_lr - k1 * self.warmup_steps
        for i in range(self.total_step):
            if i <= self.warmup_steps:
                lr.append(k1 * i + b1)
            else:
                lr.append(eta_min + (self.base_lr - eta_min) * (1 + math.cos((math.pi * i) / T_max)) / 2)
        return lr

    def Ploy_lr(self):
        eta_min = self.min_lr
        lr = []
        k1 = (self.base_lr - 1.0e-8) / self.warmup_steps
        b1 = self.base_lr - k1 * self.warmup_steps

        for i in range(self.total_step):
            if i <= self.warmup_steps:
                lr.append(k1 * i + b1)
            else:
                lr.append(eta_min + self.base_lr * (1 - ((i - self.warmup_steps) / self.total_step)) ** 3)
        return lr






class WarmUp_StepLR(BaseLR):
    def __init__(self, base_lr, total_steps, warmup_ratio, step_size, gamma=0.5):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)

        self.k = (base_lr - 1.0e-8) / (self.warmup_steps - 1)
        self.b = base_lr - self.warmup_steps * self.k

        self.gamma = gamma
        self.linear_lr_step = int((self.total_steps - self.warmup_steps) / step_size)

    def get_lr(self, cur_iter):
        # warm up
        if cur_iter <= self.warmup_steps:
            return self.k * cur_iter + self.b
        else:  # LR Phased Halving
            if (cur_iter - self.warmup_steps) % self.linear_lr_step == 0:
                self.base_lr = self.base_lr * self.gamma
            return self.base_lr





if __name__ == "__main__":
    """
    lr_policy = WarmUp_LR(base_lr=2.5e-4, total_step=100000, warmup_ratio=0.01, Types="ploy", min_lr=1.0e-7)
    lr = []
    for i in range(100000):
        lr.append(lr_policy.get_lr(i))
    print(lr[0])
    print(lr[-1])
    plt.plot(lr)
    plt.show()
    """
    total = 20000
    lr_policy = WarmUp_LR(base_lr=2.0e-4, total_steps=total,
                          warmup_ratio=0.01, Types="ploy")
    lr = []
    for i in range(1, total):
        lr.append(lr_policy.get_lr(i))
    print(lr[0])
    print(lr[-1])
    plt.plot(lr)
    plt.show()
