import matplotlib.pyplot as plt 

episodes, actor_loss, critic_loss = [], [], []
with open("actor_critic_loss.txt", "r") as f:
    for line in f:
        left, right = line.split(":")
        actor_loss_mean, critic_loss_mean = right.split()
        actor_loss_mean = float(actor_loss_mean.split("=")[1])
        critic_loss_mean = float(critic_loss_mean.split("=")[1])
        ep = int(left.replace("Episode", "").strip())
        actor_loss_mean = float(f"{actor_loss_mean:.2f}")
        critic_loss_mean = float(f"{critic_loss_mean:.2f}")
        episodes.append(ep)
        actor_loss.append(actor_loss_mean)
        critic_loss.append(critic_loss_mean)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(episodes, actor_loss, color="b", label="actor loss mean")
plt.title("Training Actor Loss Mean")
plt.xlabel("Episode")
plt.ylabel("actor loss mean")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(episodes, critic_loss, color="r", label="critic loss mean")
plt.title("Training Critic Loss Mean")
plt.xlabel("Episode")
plt.ylabel("critic loss mean")
plt.grid(True)

plt.tight_layout()

plt.savefig("ai_project_loss_result.png")
plt.show()
