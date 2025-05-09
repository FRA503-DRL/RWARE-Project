# import pandas as pd
# import matplotlib.pyplot as plt

# # โหลดข้อมูล CSV
# df = pd.read_csv("save_model_IAC_1/training_log.csv")  # เปลี่ยนชื่อไฟล์ตามจริง

# # แปลง Total Rewards จาก string → list → sum (optional)
# import ast
# # df["Total_Reward_Sum"] = df["Avg Reward"].apply(lambda x: sum(ast.literal_eval(x)))

# # พล็อตกราฟ
# plt.figure(figsize=(16, 8))

# # # กราฟ 1: Total Rewards (รวมของทุก agent)
# # plt.subplot(2, 2, 1)
# # plt.plot(df["Episode"], df["Total_Reward_Sum"])
# # plt.title("Total Reward (Sum of All Agents)")
# # plt.xlabel("Episode")
# # plt.ylabel("Total Reward")

# # กราฟ 2: Average Reward
# # plt.subplot(2, 2, 2)
# plt.plot(df["Episode"], df["Avg Reward"])
# plt.title("Average Reward")
# plt.xlabel("Episode")
# plt.ylabel("Avg Reward")

# # # กราฟ 3: Collision Rate
# # plt.subplot(2, 2, 3)
# # plt.plot(df["Episode"], df["Collision Rate"])
# # plt.title("Collision Rate")
# # plt.xlabel("Episode")
# # plt.ylabel("Rate")

# # # กราฟ 4: Time Taken
# # plt.subplot(2, 2, 4)
# # plt.plot(df["Episode"], df["Time Taken"])
# # plt.title("Episode Duration")
# # plt.xlabel("Episode")
# # plt.ylabel("Seconds")

# plt.tight_layout()
# plt.show()




import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("save_model_seac_2/training_log.csv")  # หรือใส่ path เต็ม

fig, axes = plt.subplots(7, 1, figsize=(14, 20), sharex=True)

axes[0].plot(df["Episode"], df["Avg Reward"], label="Avg Reward", color="black")
axes[0].set_ylabel("Avg")
axes[0].set_title("Average Reward per Episode")
axes[0].grid(True)
axes[0].legend()

reward_components = [
    ("WalkToWrongShelf(-0.05)", "blue"),
    ("PickRequested(+0.5)", "orange"),
    ("PickUnrequired(-0.5)", "brown"),
    ("ReturnAfterDelivery(+0.5)", "green"),
    ("DeliverToGoal(+3.0)", "red"),
    ("DropWrong(-0.3)", "purple")
]
# WalkToWrongShelf(-0.05),PickRequested(+0.5),PickUnrequired(-0.5),ReturnAfterDelivery(+0.5),DeliverToGoal(+3.0),DropWrong(-0.3)

for i, (col, color) in enumerate(reward_components, start=1):
    axes[i].plot(df["Episode"], df[col], label=col, color=color)
    axes[i].set_ylabel("Count")
    axes[i].set_title(col)
    axes[i].grid(True)
    axes[i].legend()

plt.xlabel("Episode")
plt.tight_layout()
plt.show()

