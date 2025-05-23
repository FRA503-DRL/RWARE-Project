import os
import time
import gymnasium as gym
import rware
import torch
from algorithm import IACAgent

# ----- Config -----
env_name = "rware-tiny-4ag-v2"
n_agents = 4
obs_dim = 75
action_dim = 5
lr = 3e-4
gamma = 0.99
hidden_dim = 64

n_test_episodes = 3
max_steps = 500

# ----- Load agents -----
agents = [IACAgent(obs_dim, action_dim, lr, gamma, hidden_dim) for _ in range(n_agents)]
for i in range(n_agents):
    agents[i].load_model(f"save_model_IAC_new_4/agent{i}_ep10000")  # <-- ใส่ ep ตามที่ train ไว้ เช่น 10000

# ----- Environment -----
env = gym.make(env_name)

# ----- Render 3 episodes -----
for episode in range(n_test_episodes):
    obs, _ = env.reset()
    total_rewards = [0.0] * n_agents

    for step in range(max_steps):
        actions = [agents[i].select_action(obs[i], eval_mode=False)[0] for i in range(n_agents)]  # ไม่มี eval_mode ใน IAC
        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        env.render()
        time.sleep(0.1)  # ให้ render ดูทัน

        for i in range(n_agents):
            reward = rewards[i] if isinstance(rewards, list) else rewards
            total_rewards[i] += reward

        obs = next_obs
        if terminated or truncated:
            break

    print(f"[TEST] Episode {episode+1} - Total Rewards: {total_rewards}")

env.close()
