import gymnasium as gym
import rware
import torch
from replay_buffer import TrajectoryBuffer
from algorithm import IACAgent

# ----- Config -----
env_name = "rware-small-3ag-v2"
n_agents = 3
obs_dim = 71  # เปลี่ยนตาม observation space ของ env (เช็คจาก env.observation_space.shape[0])
action_dim = 5  # เปลี่ยนตาม action space ของ env (เช็คจาก env.action_space.n)
buffer_capacity = 5000
batch_size = 32
max_episodes = 1000
max_steps = 500
gamma = 0.99
lr = 1e-3

# ----- Environment -----
env = gym.make(env_name)

print(env.observation_space[0].shape)  # obs_dim ของ agent ตัวแรก
print(env.action_space[0].n)           # action_dim ของ agent ตัวแรก

# ----- Agents & Buffers -----
agents = [IACAgent(obs_dim, action_dim, lr=lr, gamma=gamma) for _ in range(n_agents)]
buffers = [TrajectoryBuffer(capacity=buffer_capacity) for _ in range(n_agents)]

# ----- Training Loop -----
for episode in range(max_episodes):
    obs = env.reset()[0]  # env.reset() → (obs, info)
    done = [False] * n_agents
    total_rewards = [0.0] * n_agents

    for step in range(max_steps):
        actions = []
        log_probs = []

        # Select actions for all agents
        for i in range(n_agents):
            action, log_prob = agents[i].select_action(obs[i])
            actions.append(action)
            log_probs.append(log_prob)

        # Step environment
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # print(terminated)
        # print(truncated)

        dones = [terminated or truncated] * n_agents # terminated agent ทุกตัวใน env เลย (Central Episode Termination)

        # Store into buffer (per agent) ใน RWARE มัก return reward เป็น list
        for i in range(n_agents):
            reward = rewards[i] if isinstance(rewards, list) else rewards
            buffers[i].push(obs[i], actions[i], log_probs[i], reward, next_obs[i], dones[i])
            total_rewards[i] += reward

        obs = next_obs

        if all(dones):
            break

    # --- Update each agent ---
    for i in range(n_agents):
        if buffers[i].size() >= batch_size:
            batch = buffers[i].sample(batch_size)
            obs_batch, action_batch, log_prob_batch, reward_batch, next_obs_batch, done_batch = batch

            # Convert trajectory into list of tuples for the agent.update() function
            trajectory = list(zip(obs_batch, action_batch, log_prob_batch, reward_batch, next_obs_batch))
            agents[i].update(trajectory)
            buffers[i].clear()

    print(f"Episode {episode+1}/{max_episodes} - Total Rewards: {total_rewards}")

env.close()
