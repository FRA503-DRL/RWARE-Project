import os
import csv
import time
import gymnasium as gym
import rware
import torch
import numpy as np
from algorithm import SEACAgent  # Make sure SEACAgent is in algorithm.py
from replay_buffer import TrajectoryBuffer
from rware.warehouse import RewardType

# -------- CONFIG --------
env_name = "rware-tiny-4ag-v2"
n_agents = 4
obs_dim = 75
action_dim = 5
max_episodes = 10000
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
lambda_seac = 1.0

# -------- SAVE DIR --------
os.makedirs("save_model_seac", exist_ok=True)

# -------- ENVIRONMENT --------
print("Using ENV:", env_name)
env = gym.make(env_name, reward_type=RewardType.TWO_STAGE)

# -------- AGENTS --------
agents = [SEACAgent(obs_dim, action_dim, lr, gamma, hidden_dim,
                    entropy_coef, grad_clip, n_step, value_coef) for _ in range(n_agents)]
buffers = [TrajectoryBuffer(capacity=buffer_capacity) for _ in range(n_agents)]

# -------- LOGGER --------
# csv_file = open("save_model_seac/training_log.csv", mode="w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["Episode", "Avg Reward", "Total Tasks", "Episode Time"])

csv_file = open("save_model_SEAC/training_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Episode", "Avg Reward", "Total Tasks", "Episode Time",
    "WalkToWrongShelf(-0.05)", "PickRequested(+1.0)", "PickUnrequired(-0.5)",
    "ReturnAfterDelivery(+1.0)", "DeliverToGoal(+3.0)", "DropWrong(-0.5)",
])

# -------- TRAIN LOOP --------
for episode in range(max_episodes):
    obs, _ = env.reset()
    total_rewards = [0.0] * n_agents
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

        for i in range(n_agents):
            f = flags[i]
            if f["walk_wrong"]:            walk_wrong_shelf     += 1
            if f["pick_requested"]:        pick_requested       += 1
            if f["pick_unrequired"]:       pick_wrong           += 1
            if f["return_after_delivery"]: return_after_delivery+= 1
            if f["deliver_to_goal"]:
                deliver_to_goal += 1
                # total_deliveries += 1
            if f["drop_wrong"]:            drop_wrong_place     += 1

            buffers[i].push(obs[i], actions[i], log_probs[i], rewards[i], next_obs[i], done[i])
            total_rewards[i] += rewards[i]

        # # -------- RENDER --------
        # if (episode + 1) % 200 == 0:
        #     env.render()
        #     time.sleep(0.1)  # ชะลอการ render ด้วยหน่วง 0.5 วินาที

        obs = next_obs

        if all(done):
            break

    # -------- SEAC UPDATE --------
    for i in range(n_agents):
        if buffers[i].size() >= batch_size:
            # Own trajectory
            batch_i = buffers[i].sample(batch_size)
            traj_i = list(zip(*batch_i))[:5]

            # Other agents' trajectories and actors
            other_trajs = []
            other_policies = []
            for j in range(n_agents):
                if j != i and buffers[j].size() >= batch_size:
                    batch_j = buffers[j].sample(batch_size)
                    traj_j = list(zip(*batch_j))[:5]
                    other_trajs.append(traj_j)
                    other_policies.append(agents[j].actor)

            # Shared Experience Update
            agents[i].update_with_shared(traj_i, other_trajs, other_policies, lambda_=lambda_seac)
            buffers[i].clear()

    # -------- SAVE CHECKPOINT --------
    if (episode + 1) % 100 == 0:
        for i in range(n_agents):
            agents[i].save_model(f"save_model_SEAC/agent{i}_ep{episode+1}")

    # -------- LOG --------
    avg_reward = sum(total_rewards) / n_agents
    total_tasks = sum(total_rewards)
    elapsed = time.time() - start_time
    
    # Print summary
    print(f"[Episode {episode+1}] Avg Reward: {avg_reward:.2f} | Total Tasks: {total_tasks:.0f} | Time: {elapsed:.2f}s")
    print(f"  → -0.05 WalkToWrongShelf: {walk_wrong_shelf}")
    print(f"  → +1.0 PickRequested: {pick_requested}")
    print(f"  → -0.5 PickUnrequired: {pick_wrong}")
    print(f"  → +1.0 ReturnAfterDelivery: {return_after_delivery}")
    print(f"  → +3.0 DeliverToGoal: {deliver_to_goal}")
    print(f"  → -0.5 DropWrong: {drop_wrong_place}")
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

