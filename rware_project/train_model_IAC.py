# import os
# import csv
# import time as timer
# import gymnasium as gym
# import rware
# import torch
# from replay_buffer import TrajectoryBuffer
# from algorithm import IACAgent

# # ----- Config -----
# env_name = "rware-tiny-4ag-v2"
# n_agents = 4
# obs_dim = 71
# action_dim = 5
# buffer_capacity = 5000
# batch_size = 32
# max_episodes = 2000
# max_steps = 500
# gamma = 0.99
# lr = 3e-4
# hidden_dim = 64
# entropy_coef=0.01
# grad_clip=0.5
# n_step=5
# value_coef=0.5

# # ----- Create save directory -----
# os.makedirs("save_model", exist_ok=True)

# # ----- Environment -----
# env = gym.make(env_name)
# print(env.observation_space[0].shape)
# print(env.action_space[0].n)

# # ----- Agents & Buffers -----
# agents = [IACAgent(obs_dim, action_dim, lr=lr, gamma=gamma, hidden_dim=hidden_dim, entropy_coef=entropy_coef, grad_clip=grad_clip, n_step=n_step, value_coef=value_coef) for _ in range(n_agents)]
# buffers = [TrajectoryBuffer(capacity=buffer_capacity) for _ in range(n_agents)]

# # ----- CSV Logger -----
# csv_file = open("save_model/training_log.csv", mode="w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["Episode", "Total Rewards", "Average Reward", "Tasks Completed", "Collision Rate", "Time Taken"])

# # ----- Collision Detection Function -----
# def detect_collisions_from_obs(obs_list):
#     positions = {}
#     collisions = [0] * len(obs_list)

#     for i, obs in enumerate(obs_list):
#         x, y = int(obs[0]), int(obs[1])  # agent's (x, y) position
#         pos = (x, y)
#         if pos in positions:
#             collisions[i] = 1
#             collisions[positions[pos]] = 1  # mark both agents as collided
#         else:
#             positions[pos] = i
#     return collisions

# # ----- Training Loop -----
# for episode in range(max_episodes):
#     obs = env.reset()[0]
#     total_rewards = [0.0] * n_agents
#     collisions = [0] * n_agents
#     episode_start_time = timer.time()

#     for step in range(max_steps):
#         actions = []
#         log_probs = []

#         for i in range(n_agents):
#             action, log_prob = agents[i].select_action(obs[i])
#             actions.append(action)
#             log_probs.append(log_prob)

#         next_obs, rewards, terminated, truncated, infos = env.step(actions)

#         if episode % 200 == 0:
#             env.render()
#             timer.sleep(0.1)

#         dones = [terminated or truncated] * n_agents

#         step_collisions = detect_collisions_from_obs(next_obs)
#         collisions = [c + sc for c, sc in zip(collisions, step_collisions)]

#         for i in range(n_agents):
#             reward = rewards[i] if isinstance(rewards, list) else rewards
#             buffers[i].push(obs[i], actions[i], log_probs[i], reward, next_obs[i], dones[i])
#             total_rewards[i] += reward

#         obs = next_obs

#         if all(dones):
#             break

#     for i in range(n_agents):
#         if buffers[i].size() >= batch_size:
#             batch = buffers[i].sample(batch_size)
#             obs_batch, action_batch, log_prob_batch, reward_batch, next_obs_batch, done_batch = batch
#             trajectory = list(zip(obs_batch, action_batch, log_prob_batch, reward_batch, next_obs_batch))
#             agents[i].update(trajectory)
#             buffers[i].clear()

#     if (episode + 1) % 100 == 0:
#         for i in range(n_agents):
#             agents[i].save_model(f"save_model/agent{i}_ep{episode+1}")

#     avg_reward = sum(total_rewards) / n_agents
#     tasks_completed = sum(total_rewards)
#     collision_rate = sum(collisions) / (max_steps * n_agents)
#     episode_time = timer.time() - episode_start_time
#     csv_writer.writerow([episode + 1, total_rewards, avg_reward, tasks_completed, collision_rate, episode_time])
#     csv_file.flush()

#     print(f"Episode {episode+1}/{max_episodes} - Total Rewards: {total_rewards}")
#     print(f"Average reward: {avg_reward}, Collision Rate: {collision_rate:.3f}")

# csv_file.close()
# env.close()


# import os
# import csv
# import time
# import gymnasium as gym
# import rware
# import torch
# import numpy as np
# from gymnasium.vector import SyncVectorEnv
# from algorithm import IACAgent

# # -------- CONFIG --------
# env_name = "rware-tiny-4ag-v2"
# n_envs = 4
# n_agents = 4
# obs_dim = 71
# action_dim = 5
# max_episodes = 2000
# max_steps = 500
# gamma = 0.99
# lr = 3e-4
# hidden_dim = 64
# entropy_coef = 0.01
# grad_clip = 0.5
# n_step = 5
# value_coef = 0.5

# # -------- SAVE DIR --------
# os.makedirs("save_model", exist_ok=True)

# class VectorRWAREWrapper(gym.Wrapper):
#     """Make RWARE compatible with SyncVectorEnv (agent-wise rewards)."""
#     def step(self, action):
#         obs, rewards, terminated, truncated, info = self.env.step(action)

#         # make sure rewards is np.array of shape (n_agents,)
#         if isinstance(rewards, list):
#             rewards = np.array(rewards, dtype=np.float32)

#         # make obs, rewards, etc. into tuple (like gym expects)
#         return obs, rewards, terminated, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return obs, info


# # -------- PARALLEL ENV --------
# def make_env():
#     def _thunk():
#         env = gym.make(env_name)
#         env = VectorRWAREWrapper(env)  # ✅ Wrap ให้ rewards ถูกต้อง
#         return env
#     return _thunk

# print("Using ENV:", env_name)  # เช็คอีกชั้น
# env = SyncVectorEnv([make_env() for _ in range(n_envs)])

# # -------- AGENTS --------
# agents = [IACAgent(obs_dim, action_dim, lr, gamma, hidden_dim,
#                    entropy_coef, grad_clip, n_step, value_coef) for _ in range(n_agents)]

# # -------- LOGGER --------
# csv_file = open("save_model/training_log.csv", mode="w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["Episode", "Avg Reward", "Total Tasks", "Episode Time"])

# # -------- TRAIN LOOP --------
# for episode in range(max_episodes):
#     obs, _ = env.reset()
#     log_probs = [[None]*n_agents for _ in range(n_envs)]
#     trajectories = [[[] for _ in range(n_agents)] for _ in range(n_envs)]
#     total_rewards = [0.0] * n_agents
#     start_time = time.time()

#     for step in range(max_steps):
#         actions = []
#         for env_idx in range(n_envs):
#             env_actions = []
#             for agent_idx in range(n_agents):
#                 a, logp = agents[agent_idx].select_action(obs[env_idx][agent_idx])
#                 env_actions.append(a)
#                 log_probs[env_idx][agent_idx] = logp
#             actions.append(env_actions)

#         next_obs, rewards, terminated, truncated, _ = env.step(actions)

#         for env_idx in range(n_envs):
#             for agent_idx in range(n_agents):
#                 reward = rewards[env_idx][agent_idx]
#                 done = terminated[env_idx] or truncated[env_idx]
#                 trajectories[env_idx][agent_idx].append(
#                     (obs[env_idx][agent_idx], actions[env_idx][agent_idx],
#                      log_probs[env_idx][agent_idx], reward, next_obs[env_idx][agent_idx])
#                 )
#                 total_rewards[agent_idx] += reward

#         obs = next_obs
#         if all(terminated) or all(truncated):
#             break

#     # -------- UPDATE AGENTS --------
#     for agent_idx in range(n_agents):
#         merged_traj = []
#         for env_idx in range(n_envs):
#             merged_traj.extend(trajectories[env_idx][agent_idx])
#         agents[agent_idx].update(merged_traj)

#     # -------- SAVE CHECKPOINT --------
#     if (episode + 1) % 100 == 0:
#         for i in range(n_agents):
#             agents[i].save_model(f"save_model/agent{i}_ep{episode+1}")

#     # -------- LOG --------
#     avg_reward = sum(total_rewards) / (n_agents * n_envs)
#     total_tasks = sum(total_rewards)
#     elapsed = time.time() - start_time
#     csv_writer.writerow([episode + 1, avg_reward, total_tasks, elapsed])
#     csv_file.flush()

#     print(f"[Episode {episode+1}] Avg Reward: {avg_reward:.2f} | Total Tasks: {total_tasks:.0f} | Time: {elapsed:.2f}s")

# csv_file.close()
# env.close()


import os
import csv
import time
import gymnasium as gym
import rware
import torch
import numpy as np
from algorithm import IACAgent
from replay_buffer import TrajectoryBuffer
from rware.warehouse import RewardType

# -------- CONFIG --------
env_name = "rware-tiny-4ag-v2"
n_agents = 4
obs_dim = 75
action_dim = 5
max_episodes = 2000
max_steps = 500
gamma = 0.99
lr = 3e-4
hidden_dim = 64
entropy_coef = 0.001    #paper 0.01
grad_clip = 0.5
n_step = 5
value_coef = 0.5
buffer_capacity = 5000
batch_size = 32

# -------- SAVE DIR --------
os.makedirs("save_model", exist_ok=True)

# -------- ENVIRONMENT --------
print("Using ENV:", env_name)

env = gym.make(env_name, reward_type=RewardType.TWO_STAGE)

# env = gym.make(env_name)

# -------- AGENTS --------
agents = [IACAgent(obs_dim, action_dim, lr, gamma, hidden_dim,
                   entropy_coef, grad_clip, n_step, value_coef) for _ in range(n_agents)]
buffers = [TrajectoryBuffer(capacity=buffer_capacity) for _ in range(n_agents)]

# -------- LOGGER --------
# csv_file = open("save_model/training_log.csv", mode="w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["Episode", "Avg Reward", "Total Tasks", "Episode Time"])

csv_file = open("save_model/training_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Episode", "Avg Reward", "Total Tasks", "Episode Time",
    "WalkToWrongShelf(-0.05)", "PickRequested(+0.5)", "PickUnrequired(-0.5)",
    "ReturnAfterDelivery(+0.5)", "DeliverToGoal(+3.0)", "DropWrong(-0.3)",
])

# -------- TRAIN LOOP --------
for episode in range(max_episodes):
    obs, _ = env.reset()

    total_rewards = [0.0] * n_agents
    total_deliveries = 0
    start_time = time.time()

    # Tracking reward types
    walk_wrong_shelf = 0
    pick_requested = 0
    return_after_delivery = 0
    deliver_to_goal = 0
    drop_wrong_place = 0
    near_goal = 0
    pick_wrong = 0

    for step in range(max_steps):
        actions = []
        log_probs = []

        for i in range(n_agents):
            action, logp = agents[i].select_action(obs[i])
            actions.append(action)
            log_probs.append(logp)

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        flags = info["reward_flags"]
        done = [terminated or truncated] * n_agents

        # Step-by-step debug (assumes reward is sum of components)
        for i in range(n_agents):
            total_rewards[i] += rewards[i]

            f = flags[i]
            if f["walk_wrong"]:            walk_wrong_shelf     += 1
            if f["pick_requested"]:        pick_requested       += 1
            if f["pick_unrequired"]:       pick_wrong           += 1
            if f["return_after_delivery"]: return_after_delivery+= 1
            if f["deliver_to_goal"]:
                deliver_to_goal += 1
                total_deliveries += 1
            if f["drop_wrong"]:            drop_wrong_place     += 1

            buffers[i].push(obs[i], actions[i], log_probs[i], rewards[i], next_obs[i], done[i])

        # -------- RENDER --------
        if (episode + 1) % 300 == 0:
            env.render()
            time.sleep(0.1)  # ชะลอการ render ด้วยหน่วง 0.5 วินาที

        obs = next_obs

        if all(done):
            break

    # -------- UPDATE AGENTS --------
    for i in range(n_agents):
        if buffers[i].size() >= batch_size:
            batch = buffers[i].sample(batch_size)
            obs_b, act_b, logp_b, rew_b, next_b, done_b = batch
            trajectory = list(zip(obs_b, act_b, logp_b, rew_b, next_b))
            agents[i].update(trajectory)
            buffers[i].clear()

    # -------- SAVE CHECKPOINT --------
    if (episode + 1) % 100 == 0:
        for i in range(n_agents):
            agents[i].save_model(f"save_model/agent{i}_ep{episode+1}")

    # -------- LOG --------
    avg_reward = sum(total_rewards) / n_agents
    total_tasks = total_deliveries
    elapsed = time.time() - start_time

    # Print summary
    print(f"[Episode {episode+1}] Avg Reward: {avg_reward:.2f} | Total Tasks: {total_tasks:.0f} | Time: {elapsed:.2f}s")
    print(f"  → -0.05 WalkToWrongShelf: {walk_wrong_shelf}")
    print(f"  → +0.5 PickRequested: {pick_requested}")
    print(f"  → -0.5 PickUnrequired: {pick_wrong}")
    print(f"  → +0.5 ReturnAfterDelivery: {return_after_delivery}")
    print(f"  → +3.0 DeliverToGoal: {deliver_to_goal}")
    print(f"  → -0.3 DropWrong: {drop_wrong_place}")
    # print(f"  → +0.1 NearGoal: {near_goal}")



    # Save to CSV
    csv_writer.writerow([
        episode + 1, avg_reward, total_tasks, elapsed,
        walk_wrong_shelf, pick_requested, pick_wrong,
        return_after_delivery, deliver_to_goal, drop_wrong_place
    ])
    csv_file.flush()

csv_file.close()
env.close()

