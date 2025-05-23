import numpy as np

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# # 1. Actor (Policy Network)
# class Actor(nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_dim):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, action_dim)

#     def forward(self, obs):
#         x = F.relu(self.fc1(obs))
#         x = F.relu(self.fc2(x))
#         return F.softmax(self.out(x), dim=-1)  # Probability distribution over actions

# # 2. Critic (Value Network)
# class Critic(nn.Module):
#     def __init__(self, obs_dim, hidden_dim):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, 1)

#     def forward(self, obs):
#         x = F.relu(self.fc1(obs))
#         x = F.relu(self.fc2(x))
#         return self.out(x).squeeze(-1)  # Scalar value V(s)

# # 3. Agent (IAC)
# class IACAgent:
#     def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99, hidden_dim=128):
#         self.actor = Actor(obs_dim, action_dim, hidden_dim)
#         self.critic = Critic(obs_dim, hidden_dim)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
#         self.gamma = gamma

#     def select_action(self, obs, eval_mode=False):
#         obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#         probs = self.actor(obs)

#         if eval_mode:
#             # เลือก action ที่มีความน่าจะเป็นสูงสุด
#             action = torch.argmax(probs, dim=-1).item()
#             return action, None  # ไม่ใช้ log_prob ใน evaluation
#         else:
#             dist = torch.distributions.Categorical(probs)
#             action = dist.sample()
#             return action.item(), dist.log_prob(action) 

#     def compute_critic_loss(self, obs, reward, next_obs):
#         obs = torch.tensor(obs, dtype=torch.float32)
#         next_obs = torch.tensor(next_obs, dtype=torch.float32)
#         reward = torch.tensor(reward, dtype=torch.float32)

#         value = self.critic(obs)
#         next_value = self.critic(next_obs).detach()
#         target = reward + self.gamma * next_value
#         loss = F.mse_loss(value, target)
#         return loss

#     def compute_actor_loss(self, log_prob, obs, reward, next_obs):
#         obs = torch.tensor(obs, dtype=torch.float32)
#         next_obs = torch.tensor(next_obs, dtype=torch.float32)
#         reward = torch.tensor(reward, dtype=torch.float32)

#         advantage = reward + self.gamma * self.critic(next_obs).detach() - self.critic(obs).detach()
#         loss = -log_prob * advantage
#         return loss

#     def update(self, trajectory):
#         # trajectory = list of (obs, action, log_prob, reward, next_obs)
#         actor_loss_total = 0
#         critic_loss_total = 0

#         for obs, action, log_prob, reward, next_obs in trajectory:
#             critic_loss = self.compute_critic_loss(obs, reward, next_obs)
#             actor_loss = self.compute_actor_loss(log_prob, obs, reward, next_obs)

#             critic_loss_total += critic_loss
#             actor_loss_total += actor_loss

#         # Update Critic
#         self.critic_optimizer.zero_grad()
#         critic_loss_total.backward()
#         self.critic_optimizer.step()

#         # Update Actor
#         self.actor_optimizer.zero_grad()
#         actor_loss_total.backward()
#         self.actor_optimizer.step()

#     def save_model(self, path_prefix):
#         torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
#         torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
#         print(f"Model saved to {path_prefix}_actor.pth and {path_prefix}_critic.pth")

#     def load_model(self, path_prefix):
#         self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth"))
#         self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth"))
#         print(f"Model loaded from {path_prefix}_actor.pth and {path_prefix}_critic.pth")


# # Updated algorithm.py with setup closer to the paper's IAC (2006.07169)
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# class Actor(nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_dim):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, action_dim)

#     def forward(self, obs):
#         x = F.relu(self.fc1(obs))
#         x = F.relu(self.fc2(x))
#         return F.softmax(self.out(x), dim=-1)

# class Critic(nn.Module):
#     def __init__(self, obs_dim, hidden_dim):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, 1)

#     def forward(self, obs):
#         x = F.relu(self.fc1(obs))
#         x = F.relu(self.fc2(x))
#         return self.out(x).squeeze(-1)

# class IACAgent:
#     def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, hidden_dim=64, entropy_coef=0.01, grad_clip=0.5):
#         self.actor = Actor(obs_dim, action_dim, hidden_dim)
#         self.critic = Critic(obs_dim, hidden_dim)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
#         self.gamma = gamma
#         self.entropy_coef = entropy_coef
#         self.grad_clip = grad_clip

#     def select_action(self, obs, eval_mode=False):
#         obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#         probs = self.actor(obs)
#         if eval_mode:
#             action = torch.argmax(probs, dim=-1).item()
#             return action, None
#         else:
#             dist = torch.distributions.Categorical(probs)
#             action = dist.sample()
#             return action.item(), dist.log_prob(action)

#     def compute_critic_loss(self, obs, reward, next_obs):
#         obs = torch.tensor(obs, dtype=torch.float32)
#         next_obs = torch.tensor(next_obs, dtype=torch.float32)
#         reward = torch.tensor(reward, dtype=torch.float32)

#         value = self.critic(obs)
#         next_value = self.critic(next_obs).detach()
#         target = reward + self.gamma * next_value
#         loss = F.mse_loss(value, target)
#         return loss

#     def compute_actor_loss(self, log_prob, obs, reward, next_obs):
#         obs = torch.tensor(obs, dtype=torch.float32)
#         next_obs = torch.tensor(next_obs, dtype=torch.float32)
#         reward = torch.tensor(reward, dtype=torch.float32)

#         advantage = reward + self.gamma * self.critic(next_obs).detach() - self.critic(obs).detach()
#         probs = self.actor(obs.unsqueeze(0))
#         dist = torch.distributions.Categorical(probs)
#         entropy = dist.entropy().mean()
#         loss = -log_prob * advantage - self.entropy_coef * entropy
#         return loss

#     def update(self, trajectory):
#         actor_loss_total = 0
#         critic_loss_total = 0

#         for obs, action, log_prob, reward, next_obs in trajectory:
#             critic_loss = self.compute_critic_loss(obs, reward, next_obs)
#             actor_loss = self.compute_actor_loss(log_prob, obs, reward, next_obs)

#             critic_loss_total += critic_loss
#             actor_loss_total += actor_loss

#         self.critic_optimizer.zero_grad()
#         critic_loss_total.backward()
#         torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
#         self.critic_optimizer.step()

#         self.actor_optimizer.zero_grad()
#         actor_loss_total.backward()
#         torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
#         self.actor_optimizer.step()

#     def save_model(self, path_prefix):
#         torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
#         torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")

#     def load_model(self, path_prefix):
#         self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth"))
#         self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth"))


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# class Actor(nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_dim):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, action_dim)

#     def forward(self, obs):
#         x = F.relu(self.fc1(obs))
#         x = F.relu(self.fc2(x))
#         return F.softmax(self.out(x), dim=-1)

# class Critic(nn.Module):
#     def __init__(self, obs_dim, hidden_dim):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, 1)

#     def forward(self, obs):
#         x = F.relu(self.fc1(obs))
#         x = F.relu(self.fc2(x))
#         return self.out(x).squeeze(-1)

# class IACAgent:
#     def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, hidden_dim=64, entropy_coef=0.01, grad_clip=0.5, n_step=5, value_coef=0.5):
#         self.actor = Actor(obs_dim, action_dim, hidden_dim)
#         self.critic = Critic(obs_dim, hidden_dim)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
#         self.gamma = gamma
#         self.entropy_coef = entropy_coef
#         self.grad_clip = grad_clip
#         self.n_step = n_step
#         self.value_coef = value_coef

#     def select_action(self, obs, eval_mode=False):
#         obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#         probs = self.actor(obs)
#         if eval_mode:
#             action = torch.argmax(probs, dim=-1).item()
#             return action, None
#         else:
#             dist = torch.distributions.Categorical(probs)
#             action = dist.sample()
#             return action.item(), dist.log_prob(action)

#     def compute_n_step_return(self, rewards, next_obs):
#         R = 0.0
#         for i, r in enumerate(reversed(rewards)):
#             R = r + self.gamma * R
#         next_obs = torch.tensor(next_obs, dtype=torch.float32)
#         next_value = self.critic(next_obs).detach()
#         return R + (self.gamma ** len(rewards)) * next_value

#     def update(self, trajectory):
#         actor_loss_total = 0
#         critic_loss_total = 0

#         for i in range(len(trajectory) - self.n_step):
#             obs_i, action_i, log_prob_i, _, _ = trajectory[i]
#             rewards = [trajectory[i + j][3] for j in range(self.n_step)]
#             next_obs = trajectory[i + self.n_step][4]

#             obs_tensor = torch.tensor(obs_i, dtype=torch.float32)
#             value = self.critic(obs_tensor)
#             target = self.compute_n_step_return(rewards, next_obs)
#             advantage = target - value.detach()

#             # critic loss
#             critic_loss_total += F.mse_loss(value, target)

#             # actor loss
#             probs = self.actor(obs_tensor.unsqueeze(0))
#             dist = torch.distributions.Categorical(probs)
#             entropy = dist.entropy().mean()
#             actor_loss = -log_prob_i * advantage - self.entropy_coef * entropy
#             actor_loss_total += actor_loss
            
#         critic_loss_total = self.value_coef * critic_loss_total
#         # update critic
#         self.critic_optimizer.zero_grad()
#         critic_loss_total.backward()
#         torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
#         self.critic_optimizer.step()

#         # update actor
#         self.actor_optimizer.zero_grad()
#         actor_loss_total.backward()
#         torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
#         self.actor_optimizer.step()

#     def save_model(self, path_prefix):
#         torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
#         torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")

#     def load_model(self, path_prefix):
#         self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth"))
#         self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth"))


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)

class IACAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, hidden_dim=64, entropy_coef=0.01, grad_clip=0.5, n_step=5, value_coef=0.5):
        self.actor = Actor(obs_dim, action_dim, hidden_dim)
        self.critic = Critic(obs_dim, hidden_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.n_step = n_step
        self.value_coef = value_coef

    def select_action(self, obs, eval_mode=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = self.actor(obs)
        if eval_mode:
            action = torch.argmax(probs, dim=-1).item()
            return action, None
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action)

    def compute_n_step_return(self, rewards, next_obs):
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        next_value = self.critic(next_obs).detach()
        return R + (self.gamma ** len(rewards)) * next_value

    # def update(self, trajectory):
    #     total_loss = 0.0

    #     for i in range(len(trajectory) - self.n_step):
    #         obs_i, action_i, log_prob_i, _, _ = trajectory[i]
    #         rewards = [trajectory[i + j][3] for j in range(self.n_step)]
    #         next_obs = trajectory[i + self.n_step][4]

    #         obs_tensor = torch.tensor(obs_i, dtype=torch.float32)
    #         value = self.critic(obs_tensor)
    #         target = self.compute_n_step_return(rewards, next_obs)
    #         advantage = target - value.detach()

    #         critic_loss = F.mse_loss(value, target)
    #         probs = self.actor(obs_tensor.unsqueeze(0))
    #         dist = torch.distributions.Categorical(probs)
    #         entropy = dist.entropy().mean()
    #         actor_loss = -log_prob_i * advantage - self.entropy_coef * entropy

    #         total_loss += actor_loss + self.value_coef * critic_loss

    #     self.optimizer.zero_grad()
    #     total_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
    #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
    #     self.optimizer.step()

    def update(self, trajectory):
        """
        trajectory: list[(obs, action, log_prob, reward, next_obs)]
        return     : mean_actor_loss, mean_critic_loss, mean_entropy
        """
        if len(trajectory) < self.n_step + 1:
            return 0.0, 0.0, 0.0          # ไม่มีข้อมูลพอ

        actor_ls, critic_ls, ent_ls = [], [], []

        for i in range(len(trajectory) - self.n_step):
            obs_i, action_i, log_prob_i, _, _ = trajectory[i]
            rewards   = [trajectory[i + j][3] for j in range(self.n_step)]
            next_obs  = trajectory[i + self.n_step][4]

            obs_t     = torch.as_tensor(obs_i,   dtype=torch.float32)
            next_t    = torch.as_tensor(next_obs,dtype=torch.float32)

            value     = self.critic(obs_t)
            target    = self.compute_n_step_return(rewards, next_obs)
            adv       = target - value.detach()

            # losses
            critic_loss = F.mse_loss(value, target)
            probs       = self.actor(obs_t.unsqueeze(0))
            dist        = torch.distributions.Categorical(probs)
            entropy     = dist.entropy().mean()
            actor_loss  = -log_prob_i * adv - self.entropy_coef * entropy

            actor_ls.append(actor_loss.item())
            critic_ls.append(critic_loss.item())
            ent_ls.append(entropy.item())

            # accumulate for backward
            total = actor_loss + self.value_coef * critic_loss
            if i == 0:
                total_loss = total
            else:
                total_loss += total

        # ---- back-prop ----
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),  self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimizer.step()

        return (float(np.mean(actor_ls)),
                float(np.mean(critic_ls)),
                float(np.mean(ent_ls)))


    def save_model(self, path_prefix):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")

    def load_model(self, path_prefix):
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth"))





class SEACAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, hidden_dim=64, entropy_coef=0.01, grad_clip=0.5, n_step=5, value_coef=0.5):
        self.actor = Actor(obs_dim, action_dim, hidden_dim)
        self.critic = Critic(obs_dim, hidden_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.n_step = n_step
        self.value_coef = value_coef

    def select_action(self, obs, eval_mode=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = self.actor(obs)
        if eval_mode:
            action = torch.argmax(probs, dim=-1).item()
            return action, None
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action)

    def compute_n_step_return(self, rewards, next_obs):
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        next_value = self.critic(next_obs).detach()
        return R + (self.gamma ** len(rewards)) * next_value

    # def update_with_shared(self, own_traj, other_trajs, other_policies, lambda_=1.0):
    #     if len(own_traj) < self.n_step + 1:
    #         return  # ยังไม่มีข้อมูลเพียงพอ ให้ข้ามไปก่อน
    
    #     losses = []

    #     for i in range(len(own_traj) - self.n_step):
    #         obs_i, action_i, log_prob_i, _, _ = own_traj[i]
    #         rewards_i = [own_traj[i + j][3] for j in range(self.n_step)]
    #         next_obs_i = own_traj[i + self.n_step][4]

    #         obs_tensor = torch.tensor(obs_i, dtype=torch.float32)
    #         value = self.critic(obs_tensor)
    #         target = self.compute_n_step_return(rewards_i, next_obs_i)
    #         advantage = target - value.detach()

    #         # On-policy loss
    #         probs = self.actor(obs_tensor.unsqueeze(0))
    #         dist = torch.distributions.Categorical(probs)
    #         entropy = dist.entropy().mean()
    #         actor_loss = -log_prob_i * advantage - self.entropy_coef * entropy
    #         critic_loss = F.mse_loss(value, target)

    #         # Shared (off-policy) experience
    #         for k, traj_k in enumerate(other_trajs):
    #             if i >= len(traj_k) - self.n_step:
    #                 continue
    #             obs_k, act_k, _, rew_k, next_obs_k = traj_k[i]
    #             obs_k_tensor = torch.tensor(obs_k, dtype=torch.float32)
    #             next_obs_k_tensor = torch.tensor(next_obs_k, dtype=torch.float32)

    #             pi_i = self.actor(obs_k_tensor.unsqueeze(0))[0][act_k]
    #             pi_k = other_policies[k](obs_k_tensor.unsqueeze(0))[0][act_k].detach() + 1e-8  # Avoid div by 0
    #             is_weight = (pi_i / pi_k).detach()

    #             value_k = self.critic(obs_k_tensor)
    #             target_k = rew_k + (self.gamma ** self.n_step) * self.critic(next_obs_k_tensor).detach()
    #             advantage_k = target_k - value_k.detach()
    #             log_pi_i_k = torch.log(pi_i + 1e-8)

    #             actor_loss += -lambda_ * is_weight * log_pi_i_k * advantage_k
    #             critic_loss += lambda_ * is_weight * F.mse_loss(value_k, target_k)

    #         # total_loss += actor_loss + self.value_coef * critic_loss
    #         losses.append(actor_loss + self.value_coef * critic_loss)

        # self.optimizer.zero_grad()
        # total_loss = torch.stack(losses).sum()
        # total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        # self.optimizer.step()

    #---------------------CHANGE SEAC---------------------------#
    # def update_with_shared(self, own_traj, other_trajs, other_policies, lambda_=1.0):
    #     if len(own_traj) < self.n_step + 1:
    #         return 0.0, 0.0, 0.0  # <-- return loss=0 เมื่อไม่ train
        
    #     actor_losses = []
    #     critic_losses = []
    #     entropies = []

    #     for i in range(len(own_traj) - self.n_step):
    #         obs_i, action_i, log_prob_i, _, _ = own_traj[i]
    #         rewards_i = [own_traj[i + j][3] for j in range(self.n_step)]
    #         next_obs_i = own_traj[i + self.n_step][4]

    #         obs_tensor = torch.tensor(obs_i, dtype=torch.float32)
    #         value = self.critic(obs_tensor)
    #         target = self.compute_n_step_return(rewards_i, next_obs_i)
    #         advantage = target - value.detach()

    #         # On-policy loss
    #         probs = self.actor(obs_tensor.unsqueeze(0))
    #         dist = torch.distributions.Categorical(probs)
    #         entropy = dist.entropy().mean()
    #         actor_loss = -log_prob_i * advantage - self.entropy_coef * entropy
    #         critic_loss = F.mse_loss(value, target)

    #         entropies.append(entropy.item())
    #         actor_losses.append(actor_loss.item())
    #         critic_losses.append(critic_loss.item())

    #         # Off-policy shared loss
    #         for k, traj_k in enumerate(other_trajs):
    #             if i >= len(traj_k) - self.n_step:
    #                 continue
    #             obs_k, act_k, _, rew_k, next_obs_k = traj_k[i]
    #             obs_k_tensor = torch.tensor(obs_k, dtype=torch.float32)
    #             next_obs_k_tensor = torch.tensor(next_obs_k, dtype=torch.float32)

    #             pi_i = self.actor(obs_k_tensor.unsqueeze(0))[0][act_k]
    #             pi_k = other_policies[k](obs_k_tensor.unsqueeze(0))[0][act_k].detach() + 1e-8
    #             is_weight = (pi_i / pi_k).detach()

    #             value_k = self.critic(obs_k_tensor)
    #             target_k = rew_k + (self.gamma ** self.n_step) * self.critic(next_obs_k_tensor).detach()
    #             advantage_k = target_k - value_k.detach()
    #             log_pi_i_k = torch.log(pi_i + 1e-8)

    #             actor_loss += -lambda_ * is_weight * log_pi_i_k * advantage_k
    #             critic_loss += lambda_ * is_weight * F.mse_loss(value_k, target_k)

    #     total_loss = actor_loss + self.value_coef * critic_loss
    #     self.optimizer.zero_grad()
    #     total_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
    #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
    #     self.optimizer.step()

    #     return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropies)

    def update_with_shared(self,
                       own_traj,
                       other_trajs,
                       other_policies,
                       lambda_=1.0):

        # ---------- ตรวจรูปแบบข้อมูล ----------
        if isinstance(own_traj, tuple):          # ← batch tensor
            obs_b, act_b, logp_b, rew_b, next_b, _ = own_traj
            batch_mode = True
            B = obs_b.size(0)
        else:                                    # ← list[transition]
            batch_mode = False
            B = len(own_traj)

            if B < self.n_step + 1:
                return 0., 0., 0.

        actor_losses, critic_losses, entropies, total_losses = [], [], [], []

        # =========================================================
        #  iterate timesteps 0 .. B-n_step-1
        # =========================================================
        for t in range(B - self.n_step):

            # ----------- ดึงข้อมูล timestep t -----------
            if batch_mode:
                o       = obs_b[t]
                a       = act_b[t].item()
                logp    = logp_b[t]
                rewards = rew_b[t : t+self.n_step]              # Tensor  (n_step,)
                next_o  = next_b[t+self.n_step]
            else:
                o, a, logp, _, _, _ = own_traj[t]
                rewards = torch.tensor([own_traj[t+k][3] for k in range(self.n_step)],
                                    dtype=torch.float32)
                next_o  = own_traj[t+self.n_step][4]

            # ---------- critic target / advantage ----------
            R = 0.0
            for r in reversed(rewards):
                R = r + self.gamma * R
            o_t      = torch.as_tensor(o, dtype=torch.float32)
            next_t   = torch.as_tensor(next_o, dtype=torch.float32)
            target   = R + (self.gamma**self.n_step) * self.critic(next_t).detach()
            value    = self.critic(o_t)
            adv      = target - value.detach()

            # ---------- on-policy (actor/critic) ----------
            probs = self.actor(o_t.unsqueeze(0))
            dist  = torch.distributions.Categorical(probs)
            ent   = dist.entropy().mean()

            a_loss = -logp * adv - self.entropy_coef * ent
            c_loss = F.mse_loss(value, target)

            actor_losses.append(a_loss.item())
            critic_losses.append(c_loss.item())
            entropies.append(ent.item())

            # ---------- shared off-policy ----------
            for k, traj_k in enumerate(other_trajs):
                # traj_k ก็เป็น tuple tensor เช่นเดียวกัน
                if isinstance(traj_k, tuple):
                    obs_k, act_k, _, rew_k, next_k, _ = traj_k
                    if t >= obs_k.size(0) - self.n_step:
                        continue
                    o_k      = obs_k[t]
                    a_k      = act_k[t].item()
                    r_k      = rew_k[t]
                    next_o_k = next_k[t+self.n_step]
                else:  # list
                    if t >= len(traj_k) - self.n_step:
                        continue
                    o_k, a_k, _, r_k, next_o_k, _ = traj_k[t]

                o_k_t  = torch.as_tensor(o_k, dtype=torch.float32)
                next_k = torch.as_tensor(next_o_k, dtype=torch.float32)

                # importance weight
                pi_i = self.actor(o_k_t.unsqueeze(0))[0][a_k]
                pi_k = other_policies[k](o_k_t.unsqueeze(0))[0][a_k].detach() + 1e-8
                rho  = (pi_i / pi_k).detach()

                val_k  = self.critic(o_k_t)
                tgt_k  = r_k + (self.gamma**self.n_step) * self.critic(next_k).detach()
                adv_k  = tgt_k - val_k.detach()
                log_pi = torch.log(pi_i + 1e-8)

                a_loss += -lambda_ * rho * log_pi * adv_k
                c_loss +=  lambda_ * rho * F.mse_loss(val_k, tgt_k)

            total_losses.append(a_loss + self.value_coef * c_loss)

        # -------- backward --------
        self.optimizer.zero_grad()
        torch.stack(total_losses).sum().backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),  self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimizer.step()

        return (np.mean(actor_losses),
                np.mean(critic_losses),
                np.mean(entropies))



    def save_model(self, path_prefix):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")

    def load_model(self, path_prefix):
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth"))

