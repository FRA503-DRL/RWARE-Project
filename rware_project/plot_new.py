# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("save_model_IAC_new_4/training_log.csv")  # หรือใส่ path เต็ม

# fig, axes = plt.subplots(7, 1, figsize=(14, 20), sharex=True)

# axes[0].plot(df["Episode"], df["Avg Reward"], label="Avg Reward", color="black")
# axes[0].set_ylabel("Avg")
# axes[0].set_title("Average Reward per Episode")
# axes[0].grid(True)
# axes[0].legend()

# reward_components = [
#     # ("WalkToWrongShelf(-0.05)", "blue"),
#     ("PickRequested(+0.1)", "orange"),
#     # ("PickUnrequired(-0.5)", "brown"),
#     ("ReturnAfterDelivery(+0.5)", "green"),
#     ("DeliverToGoal(+4.0)", "red"),
#     # ("DropWrong(-0.3)", "purple")
# ]
# # WalkToWrongShelf(-0.05),PickRequested(+0.5),PickUnrequired(-0.5),ReturnAfterDelivery(+0.5),DeliverToGoal(+3.0),DropWrong(-0.3)

# for i, (col, color) in enumerate(reward_components, start=1):
#     axes[i].plot(df["Episode"], df[col], label=col, color=color)
#     axes[i].set_ylabel("Count")
#     axes[i].set_title(col)
#     axes[i].grid(True)
#     axes[i].legend()

# plt.xlabel("Episode")
# plt.tight_layout()
# plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("save_model_IAC_new_4/training_log.csv")

# # ---- เปลี่ยนเป็น 7 แถว ----
# fig, axes = plt.subplots(7, 1, figsize=(14, 26), sharex=True)

# # 0) Avg reward
# axes[0].plot(df["Episode"], df["Avg Reward"], color="black")
# axes[0].set_ylabel("Avg R")
# axes[0].set_title("Average Reward per Episode")
# axes[0].grid(True)

# # 1-3) reward components
# reward_components = [
#     ("PickRequested(+0.1)",      "orange"),
#     ("ReturnAfterDelivery(+0.5)","green"),
#     ("DeliverToGoal(+4.0)",      "red"),
# ]
# for i, (col, color) in enumerate(reward_components, start=1):
#     axes[i].plot(df["Episode"], df[col], label=col, color=color)
#     axes[i].set_ylabel("Count")
#     axes[i].set_title(col)
#     axes[i].grid(True)
#     axes[i].legend()

# # 4) Actor Loss
# axes[4].plot(df["Episode"], df["ActorLoss"], color="blue")
# axes[4].set_ylabel("Loss")
# axes[4].set_title("Actor Loss")
# axes[4].grid(True)

# # 5) Critic Loss
# axes[5].plot(df["Episode"], df["CriticLoss"], color="purple")
# axes[5].set_ylabel("Loss")
# axes[5].set_title("Critic Loss")
# axes[5].grid(True)

# # 6) Entropy
# axes[6].plot(df["Episode"], df["Entropy"], color="teal")
# axes[6].set_ylabel("Entropy")
# axes[6].set_title("Policy Entropy")
# axes[6].grid(True)

# plt.xlabel("Episode")
# plt.tight_layout()
# plt.show()


#-------------------------------------------------------------------------------#
# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("save_model_seac/training_log.csv")

# # --------- กำหนดหน้าตา Rolling ---------
# win = 200                   # ขนาดหน้าต่าง (episode) ปรับได้
# min_periods = 10            # ให้โชว์ค่าเฉลี่ยต้องมีอย่างน้อยกี่จุด

# # เตรียมคอลัมน์ rolling (ใช้ method mean() บน pandas.Series)
# df["AvgRewardSmooth"]  = df["Avg Reward"].rolling(win, min_periods=min_periods).mean()

# # กรองศูนย์ก่อนคำนวณ (loss=0 มักเป็น ep ที่ไม่ update)
# mask_act   = df["ActorLoss"]  > 0
# mask_cri   = df["CriticLoss"] > 0
# mask_ent   = df["Entropy"]    > 0

# df["ActorSmooth"]   = df["ActorLoss"].where(mask_act).rolling(win, min_periods=min_periods).mean()
# df["CriticSmooth"]  = df["CriticLoss"].where(mask_cri).rolling(win, min_periods=min_periods).mean()
# df["EntropySmooth"] = df["Entropy"].where(mask_ent).rolling(win, min_periods=min_periods).mean()

# # --------- Plot ---------
# fig, axes = plt.subplots(7, 1, figsize=(14, 26), sharex=True)

# # 0) Avg reward (raw + smooth)
# axes[0].plot(df["Episode"], df["Avg Reward"],  color="lightgray", linewidth=0.8, label="raw")
# axes[0].plot(df["Episode"], df["AvgRewardSmooth"], color="black", linewidth=2, label=f"MA({win})")
# axes[0].set_title("Average Reward per Episode")
# axes[0].set_ylabel("Avg R"); axes[0].legend(); axes[0].grid(True)

# # 1-3) reward components (ใส่ rolling เช่นเดียวกันถ้าต้องการ)
# reward_components = [
#     ("PickRequested(+0.1)", "orange"),
#     ("ReturnAfterDelivery(+0.5)", "green"),
#     ("DeliverToGoal(+4.0)", "red"),
# ]
# for i, (col, color) in enumerate(reward_components, start=1):
#     axes[i].plot(df["Episode"], df[col], label=col, color=color, alpha=0.3, linewidth=0.7)
#     axes[i].plot(df["Episode"],
#                  df[col].rolling(win, min_periods=min_periods).mean(),
#                  color=color, linewidth=2)
#     axes[i].set_ylabel("Count"); axes[i].set_title(col)
#     axes[i].grid(True); axes[i].legend()

# # 4) Actor loss
# axes[4].plot(df["Episode"], df["ActorLoss"], color="royalblue", alpha=0.3, linewidth=0.7)
# axes[4].plot(df["Episode"], df["ActorSmooth"], color="blue", linewidth=2, label=f"MA({win})")
# axes[4].set_ylabel("Loss"); axes[4].set_title("Actor Loss")
# axes[4].grid(True); axes[4].legend()

# # 5) Critic loss
# axes[5].plot(df["Episode"], df["CriticLoss"], color="orchid", alpha=0.3, linewidth=0.7)
# axes[5].plot(df["Episode"], df["CriticSmooth"], color="purple", linewidth=2, label=f"MA({win})")
# axes[5].set_ylabel("Loss"); axes[5].set_title("Critic Loss")
# axes[5].grid(True); axes[5].legend()

# # 6) Entropy
# axes[6].plot(df["Episode"], df["Entropy"], color="teal", alpha=0.3, linewidth=0.7)
# axes[6].plot(df["Episode"], df["EntropySmooth"], color="darkcyan", linewidth=2, label=f"MA({win})")
# axes[6].set_ylabel("Entropy"); axes[6].set_title("Policy Entropy")
# axes[6].grid(True); axes[6].legend()

# plt.xlabel("Episode")
# plt.tight_layout()
# plt.show()




#-------------------------------------------------------------------------#
import os  
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("save_model_seac_old_2/training_log.csv")

# ---------- rolling window ----------
win = 200
min_p = 10
df["AvgRewardSmooth"] = df["Avg Reward"].rolling(win, min_periods=min_p).mean()

reward_cols = [
    ("PickRequested(+0.1)",      "orange"),
    ("PickUnrequired(-0.05)",    "goldenrod"),
    ("ReturnAfterDelivery(+0.5)","green"),
    ("DeliverToGoal(+1.0)",      "red"),
    ("DropWrong(-0.1)",          "brown"),
]

for col, _ in reward_cols:
    df[f"{col}_smooth"] = df[col].rolling(win, min_periods=min_p).mean()

# ---- mask zero loss ----
mask_act = df["ActorLoss"]  > 0
mask_cri = df["CriticLoss"] > 0
mask_ent = df["Entropy"]    > 0
df["ActorSmooth"]   = df["ActorLoss"].where(mask_act).rolling(win,min_p).mean()
df["CriticSmooth"]  = df["CriticLoss"].where(mask_cri).rolling(win,min_p).mean()
df["EntropySmooth"] = df["Entropy"].where(mask_ent).rolling(win,min_p).mean()

# ---------- plotting ----------
fig, axes = plt.subplots(9, 1, figsize=(14, 32), sharex=True)

# 0) Avg reward
axes[0].plot(df["Episode"], df["Avg Reward"], color="lightgray", lw=0.8)
axes[0].plot(df["Episode"], df["AvgRewardSmooth"], color="black", lw=2,
             label=f"MA({win})")
axes[0].set_title("Average Reward per Episode"); axes[0].set_ylabel("Avg R")
axes[0].grid(True); axes[0].legend()

# 1-5) reward components
for idx, (col, color) in enumerate(reward_cols, start=1):
    axes[idx].plot(df["Episode"], df[col],             color=color,
                   alpha=0.3, lw=0.7, label=col)
    axes[idx].plot(df["Episode"], df[f"{col}_smooth"], color=color, lw=2,
                   label=f"MA({win})")
    axes[idx].set_ylabel("Count"); axes[idx].set_title(col)
    axes[idx].grid(True); axes[idx].legend()

# 6) Actor loss
axes[6].plot(df["Episode"][mask_act], df["ActorLoss"][mask_act],
             color="royalblue", alpha=0.3, lw=0.7)
axes[6].plot(df["Episode"], df["ActorSmooth"], color="blue", lw=2,
             label=f"MA({win})")
axes[6].set_ylabel("Loss"); axes[6].set_title("Actor Loss")
axes[6].grid(True); axes[6].legend()

# 7) Critic loss
axes[7].plot(df["Episode"][mask_cri], df["CriticLoss"][mask_cri],
             color="orchid", alpha=0.3, lw=0.7)
axes[7].plot(df["Episode"], df["CriticSmooth"], color="purple", lw=2,
             label=f"MA({win})")
axes[7].set_ylabel("Loss"); axes[7].set_title("Critic Loss")
axes[7].grid(True); axes[7].legend()

# 8) Entropy
axes[8].plot(df["Episode"][mask_ent], df["Entropy"][mask_ent],
             color="teal", alpha=0.3, lw=0.7)
axes[8].plot(df["Episode"], df["EntropySmooth"], color="darkcyan", lw=2,
             label=f"MA({win})")
axes[8].set_ylabel("Entropy"); axes[8].set_title("Policy Entropy")
axes[8].grid(True); axes[8].legend()


plt.xlabel("Episode")
plt.tight_layout()

# ---------- save figure ----------
os.makedirs("figs", exist_ok=True)            # โฟลเดอร์เก็บรูป
fig.savefig("figs/training_curves_SEAC_old_2.png",
            dpi=300, bbox_inches="tight")     # ปรับชื่อไฟล์ตามต้องการ

plt.show()
