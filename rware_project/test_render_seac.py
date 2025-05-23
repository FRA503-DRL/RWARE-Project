import os
import time
import gymnasium as gym
import rware
import torch
from algorithm import SEACAgent  # ใช้ SEACAgent เพราะเทรนด้วย SEAC

# ----- Config -----
env_name = "rware-tiny-4ag-v2"
n_agents = 4
obs_dim = 75
action_dim = 5
lr = 3e-4
gamma = 0.99
hidden_dim = 64
entropy_coef = 0.01
grad_clip = 0.5
n_step = 5
value_coef = 0.5

# ----- Load agents -----
agents = [SEACAgent(obs_dim, action_dim, lr, gamma, hidden_dim, entropy_coef, grad_clip, n_step, value_coef) for _ in range(n_agents)]
for i in range(n_agents):
    agents[i].load_model(f"save_model_seac_lamda_1.0/agent{i}_ep30000")  # <-- โหลด checkpoint 30000

# ----- Environment -----
env = gym.make(env_name)

# ----- Render 3 episodes -----
n_test_episodes = 3
max_steps = 500

for episode in range(n_test_episodes):
    obs, _ = env.reset()
    total_rewards = [0.0] * n_agents

    for step in range(max_steps):
        actions = [agents[i].select_action(obs[i], eval_mode=False)[0] for i in range(n_agents)]
        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        env.render()
        time.sleep(0.1)  # ชะลอให้ดูภาพ render ทัน
        
        for i in range(n_agents):
            reward = rewards[i] if isinstance(rewards, list) else rewards
            total_rewards[i] += reward

        obs = next_obs
        if terminated or truncated:
            break

    print(f"[TEST] Episode {episode+1} - Total Rewards: {total_rewards}")

env.close()
