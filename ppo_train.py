import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import keyboard
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

import gymnasium as gym

#from realenv import RealEnv

load_flag = False
# 策略网络actor
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound=2):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        #生成分布期望
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        #生成分布标准差
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        #动作幅度阈值
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))   # softplus是relu函数的平滑版本
        return mu,std

# 价值网络critic
class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
        return x
#PPO类
class PPO:
    def __init__(self, state_dim,hidden_dim, action_dim, actor_lr, critic_lr, gamma,lmbda, K_epochs, eps_clip, device):
        #实例化actor和critic网络
        self.pi = PolicyNet(state_dim, hidden_dim, action_dim).to(device)    #新策略网络
        self.old_pi = PolicyNet(state_dim, hidden_dim, action_dim).to(device)   #旧策略网络
        self.v = ValueNet(state_dim, hidden_dim).to(device)
        self.old_v = ValueNet(state_dim, hidden_dim).to(device) #旧价值网络，计算更新前后差别

        if load_flag is True:
            self.load()
        
 
        #定义优化器
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=actor_lr)
        self.v_optimizer = optim.Adam(self.v.parameters(), lr=critic_lr)

        self.step = 0      #更新步数
        self.gamma = gamma   #折扣因子
        self.lmbda = lmbda  # GAE缩放系数
        self.K_epochs = K_epochs  #更新actor网络的次数
        self.eps_clip = eps_clip  # clip参数
        self.device = device
    
    #选择动作
    def take_action(self, state):
        with torch.no_grad():
            mu, std = self.old_pi(state)
            dist = torch.distributions.normal.Normal(mu,std)  #构建正态分布
            action = dist.sample()  #采样动作
        return action.item()
    
    #更新网络
    def update(self, transitions):
        self.step += 1
        #提取数据集
        states = torch.tensor(transitions['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions['actions']).to(self.device).view(-1, 1)
        rewards = torch.tensor(transitions['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(transitions['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transitions['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        for _ in range(K_epochs):
            with torch.no_grad():
                #td目标
                td_target = rewards + self.gamma*self.old_v(next_states)*(1-done)
                #计算旧策略分布
                mu, std = self.old_pi(states)
                old_dist = torch.distributions.normal.Normal(mu,std)
                log_old_prob = old_dist.log_prob(actions)
                #td误差
                td_error = rewards + self.gamma*self.v(next_states)*(1-done) - self.v(states)
                td_error = td_error.detach().numpy()
                #GAE计算
                advantage_list = []
                adv = 0.0
                for td in td_error[::-1]:
                    adv = adv*self.gamma*self.lmbda + td[0]
                    advantage_list.append(adv)
                #正序
                advantage_list.reverse()
                advantage_list = torch.tensor(advantage_list,dtype=torch.float).reshape(-1,1)
            
            #新策略概率分布
            mu, std = self.pi(states)
            new_dist = torch.distributions.normal.Normal(mu,std)
            log_new_prob = new_dist.log_prob(actions)
            ratio = torch.exp(log_new_prob - log_old_prob)  #新旧概率分布比率
            #ppo目标函数两个子项
            sub1 = ratio * advantage_list
            sub2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)*advantage_list
            #策略更新
            loss_pi = -torch.min(sub1,sub2).mean()
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()

            #价值网络更新
            loss_v = F.mse_loss(td_target.detach(),self.v(states))
            self.v_optimizer.zero_grad()
            loss_v.backward()
            self.v_optimizer.step()
        #更新两个旧网络参数
        self.old_pi.load_state_dict(self.pi.state_dict())
        self.old_v.load_state_dict(self.v.state_dict())
    #保存模型
    def save(self, path):
        torch.save(self.pi.state_dict(), path + 'actor.pth')
        torch.save(self.v.state_dict(), path + 'critic.pth')
    #加载模型
    def load(self, path):
        try:
            self.pi.load_state_dict(torch.load(path + 'actor.pth'))
            self.v.load_state_dict(torch.load(path + 'critic.pth'))
        except:
            print("Loading is not complete!")

def get_reward(is_overturn, state, done):
    # 给定状态和动作，计算奖励
    if is_overturn and done:
        #倾倒负奖励
        return -100
    elif not is_overturn and done:
        #成功完成奖励
        return 100
    else:
        #俯仰角越小越好
        return -0.1*abs(state)


#创建保存模型的文件夹
if not os.path.exists('./models'):
    os.makedirs('./models')


rate = rclpy.Rate(10)
file_name = 'PPO_model'
writer = SummaryWriter()
env = RealEnv("env",environment_dim = 1)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
gamma = 0.99
lmbda = 0.95
K_epochs = 8    #一组训练轨迹的训练次数
eps_clip = 0.2   #clip参数
episode = 1000

actor_lr = 3e-4
critic_lr = 1e-3
return_list = []

state_dim = 1  #状态维度：俯仰角
action_dim = 1 #动作维度： 机器人姿态电机转速
hidden_dim = 64  #隐藏层维度

net = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, lmbda, K_epochs, eps_clip, device)

#训练
for i_episode in range(1000):
    #初始化环境
    state,_ = env.reset()
    done = False
    episode_reward = 0

    transitions_dir = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }

    while not done:
        #选择动作
        action = net.take_action(torch.tensor([state], dtype=torch.float))
        #执行动作
        env.action_publish(action)
        #rate.sleep()
        next_state, is_overturn, done = env.get_state()
        reward = get_reward(is_overturn,next_state,done)

        transitions_dir['states'].append(state)
        transitions_dir['actions'].append(action)
        transitions_dir['next_states'].append(next_state)
        transitions_dir['rewards'].append(reward)
        transitions_dir['dones'].append(done)

        state = next_state
        episode_reward += reward
    
    # 保存每个回合的return
    return_list.append(episode_reward)
    writer.add_scalar('return', episode_reward, i_episode)


    # 打印回合信息
    print(f'iter:{i_episode}, return:{np.mean(return_list[-10:])}')

    #更新网络
    net.update(transitions_dir)
    net.save(path='./models/')

    #人为重置机器人状态，重置好了按下回车
    print('waitting for reset......')
    keyboard.wait('ENTER')
