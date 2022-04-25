
#### 运行环境及绘图工具
所需要的python3 环境 ： Keras, Tensorflow, Numpy, Scipy and etc. 
绘图 ：matlab

#### 各个文件的含义
`./DRL_for_DDBC/codebook/codebook.mat` the file for saving the codebook matrix used in current simulation program.

`./DRL_for_DDBC/data` the folder to save the simulation results of the four schemes.

`./DRL_for_DDBC/rates` the folder to save the achievable rate of each cell within the simulation process

`./DRL_for_DDBC/base_station.py` the simulator of the base station

`./DRL_for_DDBC/cellular_network.py` the simulator of the cellular network

`./DRL_for_DDBC/channel.py` the simulator of the channels

`./DRL_for_DDBC/config.py` the file to save the configuration of the current simulation program

`./DRL_for_DDBC/data_process.py` the python script to plot the simulation results of the current simulation program

`./DRL_for_DDBC/dqn_for_singleagent.py` the DQN agent at each BS

`./DRL_for_DDBC/drl.py` the DTDE DRL-based scheme

`./DRL_for_DDBC/export_locations.py` export the locations of BSs and UEs as `.mat` files

`./DRL_for_DDBC/fp_algorithm.py` the ideal FP approach

`./DRL_for_DDBC/functions.py` some extra functions requied in the simulation

`./DRL_for_DDBC/greedy.py` the greedy scheme

`./DRL_for_DDBC/neural_network.py` the file to save the configurations of the neural network in DQN

`./DRL_for_DDBC/random_choose.py` the random scheme

`./DRL_for_DDBC/user_equipment.py` the simulator of the user equipments

