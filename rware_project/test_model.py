import os
import csv
import time as timer
import gymnasium as gym
import rware
import torch
from algorithm import IACAgent

# ----- Config -----
env_name = "rware-small-3ag-v2"
n_agents = 3
obs_dim = 71
action_dim = 5
max_episodes = 100  # จำนวน test episodes
max_steps = 500
lr = 1e-3
gamma = 0.99
hidden_dim = 128

# ----- Load agents -----
agents = [IACAgent(obs_dim, action_dim, lr=lr, gamma=gamma, hidden_dim=hidden_dim) for _ in range(n_agents)]
for i in range(n_agents):
    agents[i].load_model(f"save_model/agent{i}_ep2000")  # ระบุ checkpoint ที่ต้องการโหลด

# ----- Environment -----
env = gym.make(env_name)

# ----- CSV Logger -----
os.makedirs("save_model", exist_ok=True)
csv_file = open("save_model/test_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Episode", "Total Rewards", "Average Reward", "Tasks Completed", "Collision Rate", "Time Taken"])

# ----- Collision Detection Function -----
def detect_collisions_from_obs(obs_list):
    positions = {}
    collisions = [0] * len(obs_list)
    for i, obs in enumerate(obs_list):
        x, y = int(obs[0]), int(obs[1])
        pos = (x, y)
        if pos in positions:
            collisions[i] = 1
            collisions[positions[pos]] = 1
        else:
            positions[pos] = i
    return collisions

# ----- Testing Loop -----
for episode in range(max_episodes):
    obs = env.reset()[0]
    total_rewards = [0.0] * n_agents
    collisions = [0] * n_agents
    episode_start_time = timer.time()

    for step in range(max_steps):
        actions = [agents[i].select_action(obs[i], eval_mode=True)[0] for i in range(n_agents)]
        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        # if episode % 10 == 0:
        #     env.render()
        #     timer.sleep(0.1)

        step_collisions = detect_collisions_from_obs(next_obs)
        collisions = [c + sc for c, sc in zip(collisions, step_collisions)]
        for i in range(n_agents):
            reward = rewards[i] if isinstance(rewards, list) else rewards
            total_rewards[i] += reward

        obs = next_obs
        if terminated or truncated:
            break

    avg_reward = sum(total_rewards) / n_agents
    tasks_completed = sum(total_rewards)
    collision_rate = sum(collisions) / (max_steps * n_agents)
    episode_time = timer.time() - episode_start_time

    csv_writer.writerow([episode + 1, total_rewards, avg_reward, tasks_completed, collision_rate, episode_time])
    csv_file.flush()

    print(f"[TEST] Episode {episode+1}/{max_episodes} - Total Rewards: {total_rewards}")
    print(f"Average reward: {avg_reward}, Collision Rate: {collision_rate:.3f}")

csv_file.close()
env.close()
