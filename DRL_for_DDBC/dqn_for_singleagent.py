""" DQN agent at each base station """

import numpy as np
import random
from neural_network import NeuralNetwork
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.optimizers import rmsprop_v2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQN:
    # hyper params
    def __init__(self,
                 n_actions=NeuralNetwork().output_ports,
                 n_features=NeuralNetwork().input_ports,
                 lr=5e-4,
                 lr_decay=1e-4,
                 reward_decay=0.5,
                 e_greedy=0.6,
                 epsilon_min=1e-2,
                 replace_target_iter=100,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_decay=1e-4):
        self.n_actions = n_actions 
        self.n_features = n_features 
        self.lr = lr #初始学习率为5e-4
        self.lr_decay = lr_decay # 学习率的衰减因子，即梯度下降时搜索的步长。decay=0时，学习率不变
        self.gamma = reward_decay # gamma值越接近于1，表示越关注于未来而不是当前的回报。（不能设置的过高，否则训练难度会增大）
        # epsilon-greedy params
        self.epsilon = e_greedy # e贪心算法的初始搜索概率为0.6
        self.epsilon_decay = e_greedy_decay # 搜索概率的衰减因子。decay=0时，搜索概率不变。
        self.epsilon_min = epsilon_min # e贪心算法的最小搜索概率为0.01

        self.replace_target_iter = replace_target_iter #目标网络参数更新的时间间隔
        self.memory_size = memory_size #经验池子的大小
        self.batch_size = batch_size # 一次输入的数据量为32

        self.loss = []
        self.accuracy = []

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        self._built_net()

    def _built_net(self):

        tar_nn = NeuralNetwork()
        eval_nn = NeuralNetwork()
        self.model1 = tar_nn.get_model()
        self.model2 = eval_nn.get_model()
        self.target_replace_op()
        # RMSProp optimizer
        # 优化梯度下降的过程，使得梯度下降不会陷入局部最优忽略全局最优
        optimizer = adam_v2.Adam(learning_rate=self.lr, decay=self.lr_decay)

        self.model2.compile(loss='mse', optimizer=optimizer) 

    def _store_transition_(self, s, a, r, s_): # 将经验存起来以助于后期的动作选择
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, r, s_))
        # print(transition, '\n')
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = np.array(transition)
        self.memory_counter += 1

    def save_transition(self, s, a, r, s_):
        self._store_transition_(s, a, r, s_)

    def choose_action(self, observation): #基于e-贪心算法进行动作选择
        # epsilon greedy
        if random.uniform(0, 1) > self.epsilon: #进行探索，计算采取这个动作会收到多少回报
            observation = observation[np.newaxis, :]
            actions_value = self.model2.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = random.randint(0, self.n_actions - 1) # 仅仅是单纯的探索
        return action

    def save(self):
        self.model2.save('data/model.h5')
        self.model2.save_weights('data/weights.h5')

    def target_replace_op(self):
        temp = self.model2.get_weights() 
        self.model1.set_weights(temp)
        print('Parameters updated')

    def learn(self):
        # 达到一定的时间，迭代够次数后更新目标网络的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()

        # sample mini-batch from experience replay
        if self.memory_counter > self.memory_size:
            sample_index = random.sample(list(range(self.memory_size)), self.batch_size)
        else:
            sample_index = random.sample(list(range(self.memory_counter)), self.batch_size)

        # mini-batch data
        batch_memory = self.memory[sample_index, :]
        # 训练后返回预测结果，即标签值。 对于model1来说，返回的是本次动作的下一个动作，对于model2说，返回的是本次动作的评价
        q_next = self.model1.predict(batch_memory[:, -self.n_features:])
        q_eval = self.model2.predict(batch_memory[:, :self.n_features])
        q_target = q_eval.copy()
        
        #生成列表，第一个参数为起点，第二个参数为终点，步长取默认值1。包前不包后
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        hist = self.model2.fit(batch_memory[:, :self.n_features], q_target, verbose=0)
        self.loss.append(hist.history['loss'][0])

        self.epsilon = max(self.epsilon / (1 + self.epsilon_decay), self.epsilon_min)
        self.learn_step_counter += 1
