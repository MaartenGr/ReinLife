import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
# learning_rate = 0.0005
# gamma = 0.98
# lmbda = 0.95
# eps_clip = 0.1
# K_epoch = 3


class PPOAgent:
    def __init__(self, input_dim, output_dim, learning_rate=0.0005, gamma=0.98, lmbda=0.95,
                 eps_clip=0.1, k_epoch=3, load_model=False):
        self.agent = PPO(input_dim, output_dim, learning_rate, gamma, lmbda,eps_clip, k_epoch)
        self.method = "PPO"
        self.load_model = load_model

        if self.load_model:
            self.agent.load_state_dict(torch.load(load_model))
            self.agent.eval()

    def get_action(self, s):
        action, prob = self.agent.get_action(s)

        if self.load_model:
            return action
        else:
            return action, prob

    def learn(self):
        self.agent.learn()

    def put_data(self, transition):
        self.agent.put_data(transition)

    def data(self):
        return self.agent.data


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate, gamma, lmbda, eps_clip, k_epoch):
        super(PPO, self).__init__()
        self.data = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.k_epoch = k_epoch

        self.fc1 = nn.Linear(input_dim, 256)
        # self.fc2 = nn.Linear(256, 256)
        self.fc_pi = nn.Linear(256, output_dim)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def learn(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.k_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def get_action(self, s):
        prob = self.pi(torch.from_numpy(s).float())
        m = Categorical(prob)
        a = m.sample().item()

        return a, prob