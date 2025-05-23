
#-------------------------------------OLD ENV----------------------------------------#

# import os
# import csv
# import time
# import gymnasium as gym
# import rware
# import torch
# import numpy as np
# from algorithm import IACAgent
# from replay_buffer import TrajectoryBuffer
# from rware.warehouse import RewardType

# # -------- CONFIG --------
# env_name = "rware-tiny-4ag-v2"
# n_agents = 4
# obs_dim = 71
# action_dim = 5
# max_episodes = 10000
# max_steps = 500
# gamma = 0.99
# lr = 3e-4
# hidden_dim = 64
# entropy_coef = 0.01    #paper 0.01
# grad_clip = 0.5
# n_step = 5
# value_coef = 0.5
# buffer_capacity = 5000
# batch_size = 32

# # -------- SAVE DIR --------
# os.makedirs("save_model", exist_ok=True)

# # -------- ENVIRONMENT --------
# print("Using ENV:", env_name)

# env = gym.make(env_name, reward_type=RewardType.TWO_STAGE)

# # env = gym.make(env_name)

# # -------- AGENTS --------
# agents = [IACAgent(obs_dim, action_dim, lr, gamma, hidden_dim,
#                    entropy_coef, grad_clip, n_step, value_coef) for _ in range(n_agents)]
# buffers = [TrajectoryBuffer(capacity=buffer_capacity) for _ in range(n_agents)]

# # -------- LOGGER --------
# csv_file = open("save_model/training_log.csv", mode="w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["Episode", "Avg Reward", "Total Tasks", "Episode Time"])

# # csv_file = open("save_model/training_log.csv", mode="w", newline="")
# # csv_writer = csv.writer(csv_file)
# # csv_writer.writerow([
# #     "Episode", "Avg Reward", "Total Tasks", "Episode Time",
# #     "WalkToWrongShelf(-0.05)", "PickRequested(+0.5)", "PickUnrequired(-0.5)",
# #     "ReturnAfterDelivery(+0.5)", "DeliverToGoal(+3.0)", "DropWrong(-0.3)",
# # ])

# # -------- TRAIN LOOP --------
# for episode in range(max_episodes):
#     obs, _ = env.reset()

#     total_rewards = [0.0] * n_agents
#     total_tasks = 0
#     total_deliveries = 0
#     start_time = time.time()

#     # Tracking reward types
#     walk_wrong_shelf = 0
#     pick_requested = 0
#     return_after_delivery = 0
#     deliver_to_goal = 0
#     drop_wrong_place = 0
#     near_goal = 0
#     pick_wrong = 0

#     for step in range(max_steps):
#         actions = []
#         log_probs = []

#         for i in range(n_agents):
#             action, logp = agents[i].select_action(obs[i])
#             actions.append(action)
#             log_probs.append(logp)

#         next_obs, rewards, terminated, truncated, info = env.step(actions)
#         # flags = info["reward_flags"]
#         done = [terminated or truncated] * n_agents

#         # Step-by-step debug (assumes reward is sum of components)
#         for i in range(n_agents):
#             total_rewards[i] += rewards[i]
#             if rewards[i] == 1:
#                 total_tasks += 1

#             # f = flags[i]
#             # if f["walk_wrong"]:            walk_wrong_shelf     += 1
#             # if f["pick_requested"]:        pick_requested       += 1
#             # if f["pick_unrequired"]:       pick_wrong           += 1
#             # if f["return_after_delivery"]: return_after_delivery+= 1
#             # if f["deliver_to_goal"]:
#             #     deliver_to_goal += 1
#             #     total_deliveries += 1
#             # if f["drop_wrong"]:            drop_wrong_place     += 1

#             buffers[i].push(obs[i], actions[i], log_probs[i], rewards[i], next_obs[i], done[i])

#         # -------- RENDER --------
#         if (episode + 1) % 1000 == 0:
#             env.render()
#             time.sleep(0.1)  # ชะลอการ render ด้วยหน่วง 0.5 วินาที

#         obs = next_obs

#         if all(done):
#             break

#     # -------- UPDATE AGENTS --------
#     for i in range(n_agents):
#         if buffers[i].size() >= batch_size:
#             batch = buffers[i].sample(batch_size)
#             obs_b, act_b, logp_b, rew_b, next_b, done_b = batch
#             trajectory = list(zip(obs_b, act_b, logp_b, rew_b, next_b))
#             agents[i].update(trajectory)
#             buffers[i].clear()

#     # -------- SAVE CHECKPOINT --------
#     if (episode + 1) % 100 == 0:
#         for i in range(n_agents):
#             agents[i].save_model(f"save_model/agent{i}_ep{episode+1}")

#     # -------- LOG --------
#     avg_reward = sum(total_rewards) / n_agents
#     # total_tasks = sum(total_rewards)
#     elapsed = time.time() - start_time
#     # แปลง total_rewards ให้เป็นตัวเลขเดียวก่อนใช้ฟอร์แมต
#     episode_reward = sum(total_rewards)          # ← รวมทุกเอเจนต์

#     # Print summary
#     print(f"[Episode {episode+1}] Avg Reward: {avg_reward:.2f} | Total Reward: {episode_reward:.2f} | Total Tasks: {total_tasks:.0f} | Time: {elapsed:.2f}s")
#     # print(f"  → -0.05 WalkToWrongShelf: {walk_wrong_shelf}")
#     # print(f"  → +0.5 PickRequested: {pick_requested}")
#     # print(f"  → -0.5 PickUnrequired: {pick_wrong}")
#     # print(f"  → +0.5 ReturnAfterDelivery: {return_after_delivery}")
#     # print(f"  → +3.0 DeliverToGoal: {deliver_to_goal}")
#     # print(f"  → -0.3 DropWrong: {drop_wrong_place}")
#     # print(f"  → +0.1 NearGoal: {near_goal}")



#     # Save to CSV
#     csv_writer.writerow([
#         episode + 1, avg_reward, total_tasks, elapsed
#     #     walk_wrong_shelf, pick_requested, pick_wrong,
#     #     return_after_delivery, deliver_to_goal, drop_wrong_place
#     ])
#     csv_file.flush()

# csv_file.close()
# env.close()






#---------------------------------------NEW ENV---------------------------------------#

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
max_episodes = 30000
max_steps = 500
gamma = 0.99
lr = 3e-4
hidden_dim = 64
entropy_coef = 0.01    #paper 0.01
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
    "WalkToWrongShelf(-0)", "PickRequested(+0.1)", "PickUnrequired(-0.05)",
    "ReturnAfterDelivery(+0.5)", "DeliverToGoal(+4.0)", "DropWrong(-0.1)",
    "ActorLoss", "CriticLoss", "Entropy",
])

# -------- TRAIN LOOP --------
for episode in range(max_episodes):
    obs, _ = env.reset()

    total_rewards = [0.0] * n_agents
    total_tasks = 0
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
            # if rewards[i] == 1:
            #     total_tasks += 1

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
        # if (episode + 1) % 1000 == 0:
        #     env.render()
        #     time.sleep(0.1)  # ชะลอการ render ด้วยหน่วง 0.5 วินาที

        obs = next_obs

        if all(done):
            break
    
    actor_loss_total, critic_loss_total, entropy_total = 0.0, 0.0, 0.0
    agent_updates = 0

    # -------- UPDATE AGENTS --------
    for i in range(n_agents):
        if buffers[i].size() >= batch_size:
            batch = buffers[i].sample(batch_size)
            obs_b, act_b, logp_b, rew_b, next_b, done_b = batch
            trajectory = list(zip(obs_b, act_b, logp_b, rew_b, next_b))
            al, cl, ent = agents[i].update(trajectory)

            actor_loss_total += al
            critic_loss_total += cl
            entropy_total += ent
            agent_updates += 1
            buffers[i].clear()
        
    # Average losses
    avg_actor_loss = actor_loss_total / agent_updates if agent_updates > 0 else 0.0
    avg_critic_loss = critic_loss_total / agent_updates if agent_updates > 0 else 0.0
    avg_entropy = entropy_total / agent_updates if agent_updates > 0 else 0.0

    # -------- SAVE CHECKPOINT --------
    if (episode + 1) % 1000 == 0:
        for i in range(n_agents):
            agents[i].save_model(f"save_model/agent{i}_ep{episode+1}")

    # -------- LOG --------
    avg_reward = sum(total_rewards) / n_agents
    # total_tasks = sum(total_rewards)
    elapsed = time.time() - start_time
    # แปลง total_rewards ให้เป็นตัวเลขเดียวก่อนใช้ฟอร์แมต
    episode_reward = sum(total_rewards)          # ← รวมทุกเอเจนต์

    # Print summary
    print(f"[Episode {episode+1}] Avg Reward: {avg_reward:.2f} | Total Reward: {episode_reward:.2f} | Total Tasks: {total_deliveries:.0f} | Time: {elapsed:.2f}s | Entropy: {avg_entropy:.3f}")
    # print(f"  → -0.05 WalkToWrongShelf: {walk_wrong_shelf}")
    print(f"  → +0.1 PickRequested: {pick_requested}")
    print(f"  → -0.05 PickUnrequired: {pick_wrong}")
    print(f"  → +0.5 ReturnAfterDelivery: {return_after_delivery}")
    print(f"  → +4.0 DeliverToGoal: {deliver_to_goal}")
    print(f"  → -0.1 DropWrong: {drop_wrong_place}")
    # print(f"  → +0.1 NearGoal: {near_goal}")



    # Save to CSV
    csv_writer.writerow([
        episode + 1, avg_reward, total_tasks, elapsed,
        walk_wrong_shelf, pick_requested, pick_wrong,
        return_after_delivery, deliver_to_goal, drop_wrong_place,
        avg_actor_loss, avg_critic_loss, avg_entropy
    ])
    csv_file.flush()

csv_file.close()
env.close()

