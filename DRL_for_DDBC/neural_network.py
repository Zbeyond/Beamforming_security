""" the neural network embeded in the DQN agent """

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from config import Config


class NeuralNetwork:

    def __init__(self, input_ports=11*Config().U+7,
                 output_ports=Config().n_actions,
                 num_neurons=(64, 32), 
                 activation_function='relu'): 

        self.input_ports = input_ports # 输入层的维度
        self.output_ports = output_ports # 输出层的维度
        self.num_neurons = num_neurons #两个隐含层，一个64*64，一个32*32
        self.activation_function = activation_function #relu作为激活函数

    def get_model(self):

        model = Sequential() #按层序搭建神经网络
        model.add(Dense(self.num_neurons[0], input_shape=(self.input_ports,), activation=self.activation_function))
        for j in range(1, len(self.num_neurons)):
            model.add(Dense(self.num_neurons[j], activation=self.activation_function))
        model.add(Dense(self.output_ports))
        return model
