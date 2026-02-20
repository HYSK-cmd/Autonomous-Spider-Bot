import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from ppo import PPOAgent

env = gym.make('LunarLanderContinuous-v3') 

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, buffer_size=512)

episode_rewards = []
actor_losses = []
critic_losses = []

print("Training starts")

for episode in range(30):
    state, info = env.reset()
    state, rollout_episode_rewards = agent.collect_rollout(env, state)
    avg_episode_reward = np.mean(rollout_episode_rewards) if len(rollout_episode_rewards) > 0 else 0.0
    actor_loss, critic_loss = agent.ppo_update()
    
    episode_rewards.append(avg_episode_reward)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)
    
    print(f"[{episode+1}/50] reward: {avg_episode_reward:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

env.close()

print("\nDone Training!")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(episode_rewards, label='Episode Reward', color='blue', marker='o')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[0].set_title('Training Reward')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(actor_losses, label='Actor Loss', color='red', marker='o')
axes[1].plot(critic_losses, label='Critic Loss', color='orange', marker='s')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training Loss')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150)
plt.show()

# ===== 영상 녹화 =====
import os
video_folder = "videos/ppo"
os.makedirs(video_folder, exist_ok=True)

# 학습된 에이전트로 영상 녹화
print("\n영상 녹화 시도 중...")
try:
    env_video = gym.make('Pendulum-v1', render_mode='rgb_array')
    # 첫 번째 에피소드만 녹화 (episode_trigger: 에피소드 인덱스가 0일 때 녹화)
    env_video = gym.wrappers.RecordVideo(env_video, video_folder=video_folder, episode_trigger=lambda x: x == 0)

    state, info = env_video.reset()
    done = False
    episode_reward = 0
    max_steps = 200  # Pendulum 최대 스텝 수

    for step in range(max_steps):
        with torch.no_grad():
            action, _, _ = agent.select_action(state)
            action = action.numpy()
        
        state, reward, terminated, truncated, info = env_video.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        if done:
            # 에피소드가 끝나면 reset하여 RecordVideo가 영상을 저장하도록 함
            state, info = env_video.reset()
            break

    # 환경 종료 (RecordVideo는 자동으로 영상을 저장함)
    env_video.close()
    print(f"✓ 영상 저장 완료: {video_folder}/")
    print(f"최종 에피소드 보상: {episode_reward:.2f}")
    
except Exception as e:
    print(f"✗ 영상 저장 실패: {type(e).__name__}: {str(e)}")
    print("  대신 에이전트 성능 테스트는 진행합니다...")
    
    try:
        env_test = gym.make('Pendulum-v1')
        state, info = env_test.reset()
        done = False
        episode_reward = 0
        max_steps = 200  # Pendulum 최대 스텝 수

        for step in range(max_steps):
            with torch.no_grad():
                action, _, _ = agent.select_action(state)
                action = action.numpy()
            
            state, reward, terminated, truncated, info = env_test.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            if done:
                break
        
        env_test.close()
        print(f"✓ 테스트 완료 | 에피소드 보상: {episode_reward:.2f}")
    except Exception as e2:
        print(f"✗ 테스트 실패: {type(e2).__name__}: {str(e2)}")