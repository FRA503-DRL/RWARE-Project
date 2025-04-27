import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. Actor (Policy Network)
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x), dim=-1)  # Probability distribution over actions

# 2. Critic (Value Network)
class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)  # Scalar value V(s)

# 3. Agent (IAC)
class IACAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99):
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = self.actor(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_critic_loss(self, obs, reward, next_obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        value = self.critic(obs)
        next_value = self.critic(next_obs).detach()
        target = reward + self.gamma * next_value
        loss = F.mse_loss(value, target)
        return loss

    def compute_actor_loss(self, log_prob, obs, reward, next_obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        advantage = reward + self.gamma * self.critic(next_obs).detach() - self.critic(obs).detach()
        loss = -log_prob * advantage
        return loss

    def update(self, trajectory):
        # trajectory = list of (obs, action, log_prob, reward, next_obs)
        actor_loss_total = 0
        critic_loss_total = 0

        for obs, action, log_prob, reward, next_obs in trajectory:
            critic_loss = self.compute_critic_loss(obs, reward, next_obs)
            actor_loss = self.compute_actor_loss(log_prob, obs, reward, next_obs)

            critic_loss_total += critic_loss
            actor_loss_total += actor_loss

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss_total.backward()
        self.actor_optimizer.step()
