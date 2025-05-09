# import rware
# print(rware.__file__)

# import gymnasium as gym
# env = gym.make("rware-tiny-4ag-v2")  # ✅ ถ้าใช้ได้ แปลว่ชื่อถูก

import gymnasium as gym

print("Available environments:")
for id in sorted(gym.registry.keys()):
    if "rware" in id:
        print(id)

