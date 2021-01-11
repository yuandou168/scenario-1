import matplotlib.pyplot as plt
import numpy as np
import math

from DQN import DeepQNetwork
from env import Env
from memory import Memory
from preprocess.XMLProcess import XML2DAG
from sklearn.preprocessing import StandardScaler
import pandas as pd
import chart_studio.plotly as py
import plotly.figure_factory as ff
import datetime, random

EPISODES = 500
MINI_BATCH = 128
MEMORY_SIZE = 10000
REPLACE_TARGET_ITER = 200
N_AGENT = 2

def run_env():                  # 算法的控制流程
    step = 0
    for episode in range(EPISODES):
        rwd = [0.0, 0.0]
        obs = env.reset()       # 重新开始探索环境
        
        while True:
            step += 1
            q_value = []
            # 探索机制开始了
            if np.random.uniform() < dqn[0].epsilon:
                for i in range(N_AGENT):
                    q_value.append(dqn[i].choose_action(obs[i]))  
                # TODO(Yuandou): 如何处理两个agent的q值?
                q_joint = []  # joint q
                for i in range(len(q_value[0])):
                    # new_q = math.sqrt(pow(q_value[0][i],2) + pow(q_value[1][i],2))
                    new_q = q_value[0][i]   # 只考虑makespan的q值
                    # new_q = q_value[1][i]   # 只考虑cost的q值
                    q_joint.append(new_q)
                action = np.argmax(q_joint)
            else:
                action = np.random.randint(0, env.n_actions - 1)
    
            obs_, reward, done = env.step(action)
            
            rwd[0] += reward[0]
            rwd[1] += reward[1]
                           
            for i in range(N_AGENT):    
                memories[i].remember(obs[i], action, reward[i], obs_[i], done[i])
                size = memories[i].pointer
                batch = random.sample(range(size), size) if size < MINI_BATCH else random.sample(range(size), MINI_BATCH)
                if step > REPLACE_TARGET_ITER and step % 5 == 0:
                    dqn[i].learn(*memories[i].sample(batch))

            obs = obs_
 
            if done[0]:   
                if episode % 1 == 0:
                    print(
                            'episode:' + str(episode) + ' steps:' + str(step) +
                            ' reward0:' + str(round(rwd[0],6)) + ' reward1:' + str(round(rwd[1],6)) +
                            ' eps_greedy0:' + str(round(dqn[0].epsilon,6)) + ' eps_greedy1:' + str(round(dqn[1].epsilon,6)), 
                            ' makespan:' + str(max(env.vm_time)),   # makespan
                            ' cost:' + str(np.sum(env.vm_cost)) # total cost
                        )    

                # 画图用的数据
                for i in range(N_AGENT):
                    rewards[i].append(rwd[i])       
                records[0].append(max(env.vm_time))     # makespan的记录值
                records[1].append(np.sum(env.vm_cost))   # cost的记录值
                records[2].append(env.strategies)   # strategy的记录值
                break
        
def plot_pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    '''Plotting process'''
    plt.plot(Xs, Ys, '.b', markersize=16, label='Non Pareto-optimal')
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y, '.r', markersize=16, label='Pareto optimal')
    plt.title('Pareto frontier selection')
    plt.xlabel("maximum completion time(makespan)")
    plt.ylabel("total cost")
    _=plt.legend(loc="upper right", numpoints=1)
    plt.savefig('./pictures/pareto_frontier_results.svg')
    plt.show() 
    return pf_X, pf_Y


if __name__ == '__main__':
    rewards = [[], []]          # makespan agent 和 cost agent的奖励值
    records = [[], [], []]          # makespan agent 和 cost agent的数值以及策略集合
    
    scaler = StandardScaler()

    env = Env(N_AGENT)
    memories = [Memory(MEMORY_SIZE) for i in range(N_AGENT)]
    memory = Memory(MEMORY_SIZE)

    dqn = [DeepQNetwork(env.n_actions,
                        env.n_features,
                        i,
                        learning_rate=0.0001,
                        replace_target_iter=REPLACE_TARGET_ITER,
                        e_greedy_increment=2e-5
                        ) for i in range(N_AGENT)]

    run_env()
    
    fig, (ax0, ax1) = plt.subplots(nrows=2)   
    ax0.grid(True)
    ax0.set_xlabel('episodes')
    ax0.set_ylabel('makespan metric')
    line01, = ax0.plot(rewards[0], color='orange', label = "rewards", linestyle='-')
    # line02, = ax0.plot(records[0], label = "records", linewidth=2)
    ax0.add_artist(ax0.legend(handles=[line01], loc='upper left'))
    # ax0.legend(handles=[line02], loc='lower left')
    
    ax1.grid(True)
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('cost metric')
    line11, = ax1.plot(rewards[1], label = "rewards", linestyle='-')
    # line12, = ax1.plot(records[1], label = "records", linewidth=2)
    ax1.add_artist(ax1.legend(handles=[line11], loc='upper left'))
    # ax1.legend(handles=[line12], loc='lower left')

    fig.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    plt.savefig ('./pictures/makespan&cost_conv_dqn.svg')
    plt.show()

    # Pareto optimal
    p = plot_pareto_frontier(records[0], records[1], maxX=False, maxY=False)
    print("pareto optimal: ", p)

    opt_index = records[0].index(p[0][0])
    print(opt_index)
    opt_strategy = records[2][opt_index]
    
    print('DQN-based MARL Gantt图========')
    c_tmp = pd.read_excel("./datasets/WSP_dataset.xlsx", sheet_name="Containers Price")
    CONTAINERS = list(c_tmp.loc[:, 'Configuration Types'])
    m_keys = [j + 1 for j in range (len(CONTAINERS))]
    j_keys = [j for j in range (5)]
    df = []

    record = []
    for k in opt_strategy:
        start_time = str (datetime.timedelta (seconds=k[2]))
        end_time = str (datetime.timedelta (seconds=k[3]))
        record.append ((k[0], k[1], [start_time, end_time]))
    print(len(record))

    for m in m_keys:
        for j in j_keys:
                for i in record:
                    if (m, j) == (i[1], i[0]):
                            df.append (dict (Task='Container %s ' % (CONTAINERS[m - 1]), Start='2020-01-16 %s' % (str(i[2][0])),
                                                        Finish='2020-01-16 %s' % (str (i[2][1])),
                                                        Resource='Workflow %s' % (j + 1)))
    fig = ff.create_gantt (df, index_col='Resource', show_colorbar=True, group_tasks=True,
                                            showgrid_x=True,
                                            title='DQN-based MARL Workflow Scheduling')
    py.plot(fig, filename='DQN-based_MARL_workflow_scheduling', world_readable=True)