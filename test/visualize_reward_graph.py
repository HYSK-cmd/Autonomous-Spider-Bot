import matplotlib.pyplot as plt 

episodes, rewards, means = [], [], []
with open("reward_log2.txt", "r") as f:
    for line in f:
        left, right = line.split(":")
        reward, mean = right.split()
        reward = float(reward.split("=")[1])
        mean = float(mean.split("=")[1])
        ep = int(left.replace("Episode", "").strip())
        reward = float(f"{reward:.2f}")
        mean = float(f"{mean:.2f}")
        episodes.append(ep)
        rewards.append(reward)
        means.append(mean)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(episodes, rewards, color="b", label="reward")
plt.title("Training Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(episodes, means, color="r", label="mean")
plt.title("Training Episode Reward Mean")
plt.xlabel("Episode")
plt.ylabel("Reward Mean")
plt.grid(True)

plt.tight_layout()

plt.savefig("graph_result.png")
plt.show()
