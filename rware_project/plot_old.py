import pandas as pd
import matplotlib.pyplot as plt

# โหลดไฟล์ CSV
df = pd.read_csv("save_model_seac_old_2/test_1000_log.csv")  # เปลี่ยนชื่อไฟล์ตามจริง

# คำนวณค่าเฉลี่ย
avg_reward_mean = df["Avg Reward"].mean()
total_tasks_mean = df["Total Tasks"].mean()

# ตั้งค่าขนาดภาพ
plt.figure(figsize=(14, 8))

# --- กราฟ 1: Avg Reward ---
plt.subplot(2, 1, 1)
plt.plot(df["Episode"], df["Avg Reward"], marker='.', linewidth=0.5, markersize=3, color='blue')
plt.title("Average Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Avg Reward")

# --- กราฟ 2: Total Tasks ---
plt.subplot(2, 1, 2)
plt.plot(df["Episode"], df["Total Tasks"], marker='.', linewidth=0.5, markersize=3, color='green')
plt.title("Total Tasks Completed per Episode")
plt.xlabel("Episode")
plt.ylabel("Tasks Completed")

# จัด layout
plt.tight_layout()
plt.show()

# พิมพ์ค่าเฉลี่ยออกมา
print(f"Average of Avg Reward across all episodes: {avg_reward_mean:.4f}")
print(f"Average of Total Tasks across all episodes: {total_tasks_mean:.4f}")
