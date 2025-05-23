import os
import csv
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

n_test_episodes = 1000
max_steps = 500

# ----- Load agents -----
agents = [IACAgent(obs_dim, action_dim, lr, gamma, hidden_dim) for _ in range(n_agents)]
for i in range(n_agents):
    agents[i].load_model(f"save_model_IAC_new_ENV4.2/agent{i}_ep30000")  # เปลี่ยนตรงนี้ตาม checkpoint ที่ train ไว้

# ----- Environment -----
env = gym.make(env_name)

# ----- CSV Logger -----
# os.makedirs("save_model_iac_test", exist_ok=True)
csv_file = open("save_model_IAC_new_ENV4.2/test_1000_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Episode", "Avg Reward", "Total Tasks", "Episode Time"])

# ----- Testing Loop -----
for episode in range(n_test_episodes):
    obs, _ = env.reset()
    total_rewards = [0.0] * n_agents
    total_tasks = 0
    start_time = time.time()

    for step in range(max_steps):
        actions = [agents[i].select_action(obs[i], eval_mode=False)[0] for i in range(n_agents)]  # ไม่มี eval_mode
        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        for i in range(n_agents):
            reward = rewards[i] if isinstance(rewards, list) else rewards
            total_rewards[i] += reward
            if reward == 4:
                total_tasks += 1

        obs = next_obs
        if terminated or truncated:
            break

    avg_reward = sum(total_rewards) / n_agents
    episode_time = time.time() - start_time

    # Save result to CSV
    csv_writer.writerow([episode + 1, avg_reward, total_tasks, episode_time])
    csv_file.flush()

    print(f"[TEST] Episode {episode+1}/{n_test_episodes} - Avg Reward: {avg_reward:.2f} | Total Tasks: {total_tasks} | Time: {episode_time:.2f}s")

csv_file.close()
env.close()
