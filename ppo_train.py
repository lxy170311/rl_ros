import torch
import rclpy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import keyboard
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

from realenv import RealEnv

load_flag = False  # 是否加载模型
anner_flag = False  # 是否退火
episode_num = 1000  # 训练的episode数

#change
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std) # 权重用正交初始化
    nn.init.constant_(layer.bias, bias_const) # 偏置用常数初始化
    return layer

# 策略网络actor
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound=50):
        super(PolicyNet, self).__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            # 生成分布期望
            layer_init(nn.Linear(hidden_dim, action_dim),std=0.01)
        )
        # 生成分布标准差
        self.actor_logstd = nn.Parameter(torch.zeros(1,action_dim))
        # 动作幅度阈值
        self.action_bound = action_bound

    def get_action(self, x,action = None):
        mu = self.actor_mean(x)
        std = self.actor_logstd.expand_as(mu)
        std = torch.exp(std)
        dist = Normal(mu, std)
        if action is None:
            action = dist.sample()  # 采样动作
        action = torch.tanh(action)  # 限制动作幅度到[-1,1]
        action = action * self.action_bound  # 动作幅度阈值
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1)


# 价值网络critic
class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_states, n_hiddens)),
            nn.ReLU(),
            layer_init(nn.Linear(n_hiddens, n_hiddens)),
            nn.ReLU(),
            layer_init(nn.Linear(n_hiddens, 1))
        )

    def get_value(self, x):
        return self.critic(x)


# PPO类
class PPO:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        gamma,
        lmbda,
        K_epochs,
        eps_clip,
        device,
    ):
        # 实例化actor和critic网络
        self.pi = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  # 策略网络
        self.old_pi = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  # 旧策略网络
        self.v = ValueNet(state_dim, hidden_dim).to(device) #价值网络
        self.old_v = ValueNet(state_dim, hidden_dim).to(device) #旧价值网络

        if load_flag is True:
            self.load()

        # 定义优化器
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=actor_lr)
        self.v_optimizer = optim.Adam(self.v.parameters(), lr=critic_lr)

        self.step = 0  # 更新步数
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE缩放系数
        self.K_epochs = K_epochs  # 更新actor网络的次数
        self.eps_clip = eps_clip  # clip参数
        self.device = device
        self.action_bound = 50  # 动作幅度阈值
        self.actor_loss = 0.0
        self.critic_loss = 0.0

        self.mini_batch_size = 64  # mini-batch大小

    
    def annear_lr(self, episode):
        #lr退火
        frac = 1.0 - episode / episode_num
        lr_pi = frac * 3e-4
        lr_v = frac * 1e-3
        for param_group in self.pi_optimizer.param_groups:
            param_group["lr"] = lr_pi
        for param_group in self.v_optimizer.param_groups:
            param_group["lr"] = lr_v


    # 更新网络
    def update(self, transitions):
        self.step += 1
        # 提取数据集
        states = torch.tensor(transitions["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions["actions"]).to(self.device).view(-1, 1)
        rewards = (
            torch.tensor(transitions["rewards"], dtype=torch.float)
            .to(self.device)
            .view(-1, 1)
        )
        next_states = torch.tensor(transitions["next_states"], dtype=torch.float).to(
            self.device
        )
        dones = (
            torch.tensor(transitions["dones"], dtype=torch.float)
            .to(self.device)
            .view(-1, 1)
        )

        for _ in range(K_epochs):
            with torch.no_grad():
                # td目标
                td_target = rewards + self.gamma * self.old_v.get_value(next_states) * (1 - dones)
                # 计算旧策略分布
                _, log_old_prob, _ = self.old_pi.get_action(states)
                # td误差
                td_error = (
                    rewards
                    + self.gamma * self.v.get_value(next_states) * (1 - dones)
                    - self.v.get_value(states)
                )
                td_error = td_error.detach().numpy()
                # GAE计算
                advantage_list = []
                adv = 0.0
                for td in td_error[::-1]:
                    adv = adv * self.gamma * self.lmbda + td
                    advantage_list.append(adv)
                # 正序
                advantage_list.reverse()
                advantage_list = torch.tensor(
                    advantage_list, dtype=torch.float
                ).reshape(-1, 1)

            # 新策略概率分布
            _, log_new_prob, entropy = self.pi.get_action(states,actions)
            ratio = torch.exp(log_new_prob - log_old_prob)  # 新旧概率分布比率
            # ppo目标函数两个子项
            sub1 = ratio * advantage_list
            sub2 = (
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantage_list
            )
            # 策略更新
            loss_pi = -torch.min(sub1, sub2).mean()
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()
            self.actor_loss = loss_pi.item()

            # 价值网络更新
            loss_v = torch.mean(F.mse_loss(td_target.detach(), self.v.get_value(states)))
            self.v_optimizer.zero_grad()
            loss_v.backward()
            self.v_optimizer.step()
            self.critic_loss = loss_v.item()
        
        
        # 更新两个旧网络参数
        self.old_pi.load_state_dict(self.pi.state_dict())
        self.old_v.load_state_dict(self.v.state_dict())

    # 保存模型
    def save(self, path):
        torch.save(self.pi.state_dict(), path + "actor.pth")
        torch.save(self.v.state_dict(), path + "critic.pth")

    # 加载模型
    def load(self, path):
        try:
            self.pi.load_state_dict(torch.load(path + "actor.pth"))
            self.v.load_state_dict(torch.load(path + "critic.pth"))
        except:
            print("Loading is not complete!")


def get_reward(is_overturn, state, done):
    # 给定状态和动作，计算奖励
    if is_overturn and done:
        # 倾倒负奖励
        return -100
    elif not is_overturn and done:
        # 成功完成奖励
        return 100
    else:
        # 俯仰角越小越好
        return -0.1 * abs(state)


# 创建保存模型的文件夹
if not os.path.exists("./models"):
    os.makedirs("./models")


rate = rclpy.Rate(10)
file_name = "PPO_model"
writer = SummaryWriter()
env = RealEnv("env", environment_dim=1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
gamma = 0.99
lmbda = 0.95
K_epochs = 8  # 一组训练轨迹的训练次数
eps_clip = 0.2  # clip参数

actor_lr = 3e-4
critic_lr = 1e-3
return_list = []

state_dim = 1  # 状态维度：俯仰角
action_dim = 1  # 动作维度： 机器人姿态电机转速
hidden_dim = 64  # 隐藏层维度

net = PPO(
    state_dim,
    hidden_dim,
    action_dim,
    actor_lr,
    critic_lr,
    gamma,
    lmbda,
    K_epochs,
    eps_clip,
    device,
)

# 训练
for i_episode in range(episode_num):
    # 初始化环境
    state, _ = env.reset()
    done = False
    episode_reward = 0

    transitions_dir = {
        "states": [],
        "actions": [],
        "next_states": [],
        "rewards": [],
        "dones": [],
    }
    if anner_flag:
        net.annear_lr(i_episode)
    while not done:
        # 选择动作
        action,log_prob,entropy = net.pi.get_action(torch.Tensor(state).to(device))
        action = action.item()
        # 执行动作
        env.action_publish(action)
        rate.sleep()
        next_state, is_overturn, done = env.get_state()
        reward = get_reward(is_overturn, next_state, done)

        transitions_dir["states"].append(state)
        transitions_dir["actions"].append(action)
        transitions_dir["next_states"].append(next_state)
        transitions_dir["rewards"].append(reward)
        transitions_dir["dones"].append(done)

        state = next_state
        episode_reward += reward

    # 保存每个回合的return
    return_list.append(episode_reward)
    writer.add_scalar("actor_loss", net.actor_loss, i_episode)
    writer.add_scalar("critic_loss", net.critic_loss, i_episode)
    writer.add_scalar("entropy", entropy.item(), i_episode)
    writer.add_scalar("return", episode_reward, i_episode)

    # 打印回合信息
    print(f"iter:{i_episode}, return:{np.mean(return_list[-10:])}")

    # 更新网络
    net.update(transitions_dir)
    net.save(path="./models/")

    # 人为重置机器人状态，重置好了按下回车
    print("waitting for reset......")
    keyboard.wait("ENTER")
